import argparse
import copy
import os
import random
import re
import string

import pandas as pd
import streamlit as st
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from loguru import logger
from st_aggrid import AgGrid, AgGridTheme, ColumnsAutoSizeMode, GridOptionsBuilder, GridUpdateMode

from autotrain import help
from autotrain.dataset import AutoTrainDataset, AutoTrainDreamboothDataset, AutoTrainImageClassificationDataset
from autotrain.params import Params
from autotrain.project import Project
from autotrain.tasks import COLUMN_MAPPING, NLP_TASKS, TABULAR_TASKS, TASK_TYPE_MAPPING, VISION_TASKS
from autotrain.utils import get_project_cost, get_user_token, user_authentication


def parse_args():
    """
    Parse command line arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description="AutoTrain app")
    parser.add_argument(
        "--task",
        type=str,
        required=False,
    )
    return parser.parse_args()


def does_repo_exist(repo_id, repo_type) -> bool:
    try:
        HfApi().repo_info(repo_id=repo_id, repo_type=repo_type)
        return True
    except RepositoryNotFoundError:
        return False


def verify_project_name(project_name, username, user_token):
    """
    Verify that the project name is valid
    :param project_name: name of the project
    :param username: username of the user
    :return: True if project name is valid, False otherwise
    """
    if project_name == "":
        st.error("Project name cannot be empty")
        return False
    if len(project_name) > 50:
        st.error("Project name cannot be longer than 50 characters")
        return False
    pattern = "^[A-Za-z0-9-]*$"
    if not re.match(pattern, project_name):
        st.error("Project name can only contain letters, numbers and hyphens")
        return False
    if user_token is None:
        st.error("You need to be logged in to create a project. Please login using `huggingface-cli login`")
        return False
    data_repo_name = f"{username}/{project_name}"
    if does_repo_exist(data_repo_name, "dataset"):
        st.error("A project with this name already exists")
        return False
    return True


def get_job_params(job_params, selected_rows, task, model_choice):
    """
    Get job parameters list of dicts for AutoTrain and HuggingFace Hub models
    :param job_params: job parameters
    :param selected_rows: selected rows
    :param task: task
    :param model_choice: model choice
    :return: job parameters list of dicts
    """
    if model_choice == "AutoTrain":
        if len(job_params) > 1:
            raise ValueError("❌ Only one job parameter is allowed for AutoTrain.")
        job_params[0].update({"task": task})
    elif model_choice == "HuggingFace Hub":
        for i in range(len(job_params)):
            job_params[i].update({"task": task})
        job_params = [job_params[i] for i in selected_rows]
    logger.info("***")
    logger.info(job_params)
    return job_params


def create_grid(jobs):
    """
    Create Aggrid for job parameters
    :param jobs: job parameters from streamlit session_state
    :return: Aggrid
    """

    df = pd.DataFrame(jobs)
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        cellStyle={"color": "black", "font-size": "12px"},
        suppressMenu=True,
        wrapHeaderText=True,
        autoHeaderHeight=True,
    )
    gb.configure_selection(
        selection_mode="multiple",
        use_checkbox=True,
        pre_selected_rows=list(range(len(df))),
        header_checkbox=True,
    )
    custom_css = {
        ".ag-header-cell-text": {"font-size": "12px", "text-overflow": "revert;", "font-weight": 700},
        # ".ag-theme-streamlit": {"transform": "scale(0.8)", "transform-origin": "0 0"},
    }
    gridOptions = gb.build()
    ag_resp = AgGrid(
        df,
        gridOptions=gridOptions,
        custom_css=custom_css,
        # allow_unsafe_jscode=True,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        theme=AgGridTheme.STREAMLIT,  # Only choices: AgGridTheme.STREAMLIT, AgGridTheme.ALPINE, AgGridTheme.BALHAM, AgGridTheme.MATERIAL
        # width='100%',
        update_mode=GridUpdateMode.MODEL_CHANGED,
        update_on="MANUAL",
        reload_data=False,
    )
    return ag_resp


def app():  # username, valid_orgs):
    st.sidebar.markdown(
        "<p style='text-align: center; font-size: 20px; font-weight: bold;'>AutoTrain Advanced</p>",
        unsafe_allow_html=True,
    )
    user_token = os.environ.get("HF_TOKEN", "")
    if len(user_token) == 0:
        user_token = get_user_token()
    if user_token is None:
        st.error(
            """You need to be logged in to create a project. Please login using `huggingface-cli login` or pass your HF token in an environment variable called `HF_TOKEN`
            \nIf you are running this app in Hugging Face Space, please [duplicate this space](https://huggingface.co/spaces/autotrain-projects/autotrain-advanced?duplicate=true) to your own user and add `HF_TOKEN` as a secret."""
        )
        return
    if len(user_token) == 0:
        return
    user_info = user_authentication(token=user_token)
    username = user_info["name"]
    user_can_pay = user_info["canPay"]

    orgs = user_info["orgs"]
    valid_orgs = [org for org in orgs if org["canPay"] is True]
    valid_orgs = [org for org in valid_orgs if org["roleInOrg"] in ("admin", "write")]
    valid_orgs = [org["name"] for org in valid_orgs]

    if user_can_pay is False and len(valid_orgs) == 0:
        st.error(
            "Please attach a payment method to your account / join an organization with a valid payment method attached to it to create a project"
        )
        return

    who_is_training = [username] + valid_orgs
    st.markdown("###### Project Info")
    random_proj_string = "".join(random.choices(string.ascii_lowercase + string.digits, k=7))
    col1, col2 = st.columns(2)
    with col1:
        autotrain_username = st.selectbox("Who is training?", who_is_training, help=help.APP_AUTOTRAIN_USERNAME)
    with col2:
        project_name = st.text_input(
            "Project name", f"autotrain-project-{random_proj_string}", help=help.APP_PROJECT_NAME
        )

    if "task" in st.session_state:
        project_type = TASK_TYPE_MAPPING[st.session_state.task]
        task = st.session_state.task
    else:
        col1, col2 = st.columns(2)
        with col1:
            project_type = st.selectbox(
                "Project Type",
                [
                    "Natural Language Processing",
                    "Computer Vision",
                    # "Tabular",
                ],
            )
        with col2:
            if project_type == "Natural Language Processing":
                task = st.selectbox("Task", list(NLP_TASKS.keys()))
            elif project_type == "Computer Vision":
                task = st.selectbox("Task", list(VISION_TASKS.keys()))
            elif project_type == "Tabular":
                task = st.selectbox("Task", list(TABULAR_TASKS.keys()))

    if task == "image_classification":
        task = "image_binary_classification"
    if task == "text_classification":
        task = "text_binary_classification"
    # st.markdown("""---""")
    st.markdown("###### Model choice")
    if task.startswith("tabular"):
        model_choice = "AutoTrain"
    elif task == "dreambooth":
        model_choice = "HuggingFace Hub"
    else:
        model_choice_label = ["AutoTrain", "HuggingFace Hub"]
        model_choice = st.selectbox("Model Choice", model_choice_label, label_visibility="collapsed")

    hub_model = None
    if model_choice == "HuggingFace Hub":
        default_hub_model = "bert-base-uncased"
        if task == "dreambooth":
            default_hub_model = "stabilityai/stable-diffusion-2-1-base"
        if task.startswith("image"):
            default_hub_model = "google/vit-base-patch16-224"
        hub_model = st.text_input("Model name", default_hub_model)
    # st.markdown("""---""")
    st.markdown("###### Data")
    if task == "dreambooth":
        number_of_concepts = st.number_input("Number of concepts", min_value=1, max_value=5, value=1)
        tabs = st.tabs([f"Concept {i + 1}" for i in range(number_of_concepts)])
        for i in range(number_of_concepts):
            with tabs[i]:
                st.text_input(f"Concept {i + 1} token", key=f"dreambooth_concept_name_{i + 1}")
                st.file_uploader(
                    f"Concept {i + 1} images",
                    key=f"dreambooth_concept_images_{i + 1}",
                    type=["png", "jpg", "jpeg"],
                    accept_multiple_files=True,
                )
        training_data = "dreambooth"
    else:
        tab1, tab2 = st.tabs(["Training", "Validation (Optional)"])
        with tab1:
            if project_type == "Computer Vision":
                training_images = st.file_uploader(
                    "Training Images", type=["zip"], help=help.APP_IMAGE_CLASSIFICATION_DATA_HELP
                )
                training_data = task
            else:
                training_data = st.file_uploader("Training Data", type=["csv", "jsonl"], accept_multiple_files=True)

        with tab2:
            if project_type == "Computer Vision":
                validation_images = st.file_uploader(
                    "Validation Images", type=["zip"], help=help.APP_IMAGE_CLASSIFICATION_DATA_HELP
                )
                validation_data = task
            else:
                validation_data = st.file_uploader(
                    "Validation Data", type=["csv", "jsonl"], accept_multiple_files=True
                )

    if "training_data" in locals() and training_data:
        if task not in ("dreambooth", "image_binary_classification"):
            st.markdown("###### Column mapping")
            # read column names
            temp_train_data = copy.deepcopy(training_data[0])
            if temp_train_data.name.endswith(".csv"):
                df = pd.read_csv(temp_train_data, nrows=0)
            elif temp_train_data.name.endswith(".jsonl"):
                df = pd.read_json(temp_train_data, lines=True, nrows=0)
            else:
                raise ValueError("Unknown file type")
            columns = list(df.columns)
            for map_idx, map_name in enumerate(COLUMN_MAPPING[task]):
                if map_name == "id" and task.startswith("tabular"):
                    st.selectbox(f"Map `{map_name}` to:", columns + [""], index=map_idx, key=f"map_{map_name}")
                else:
                    st.selectbox(f"Map `{map_name}` to:", columns, index=map_idx, key=f"map_{map_name}")

        st.sidebar.markdown("### Parameters")
        params = Params(task=task, training_type="autotrain" if model_choice == "AutoTrain" else "hub_model")
        params = params.get()

        if model_choice == "AutoTrain":
            st.sidebar.markdown("Hyperparameters are selected automagically for AutoTrain models")

        for key, value in params.items():
            if value.STREAMLIT_INPUT == "selectbox":
                st.sidebar.selectbox(value.PRETTY_NAME, value.CHOICES, 0, key=f"params__{key}")
            elif value.STREAMLIT_INPUT == "number_input":
                try:
                    step = value.STEP
                except AttributeError:
                    step = None
                try:
                    _format = value.FORMAT
                except AttributeError:
                    _format = None
                st.sidebar.number_input(
                    value.PRETTY_NAME,
                    value.MIN_VALUE,
                    value.MAX_VALUE,
                    value.DEFAULT,
                    step=step,
                    format=_format,
                    key=f"params__{key}",
                )
            elif value.STREAMLIT_INPUT == "slider":
                st.sidebar.slider(
                    value.PRETTY_NAME,
                    value.MIN_VALUE,
                    value.MAX_VALUE,
                    value.DEFAULT,
                    key=f"params__{key}",
                )
        if model_choice == "AutoTrain":
            st.session_state.jobs = []
            st.session_state.jobs.append(
                {k[len("params__") :]: v for k, v in st.session_state.items() if k.startswith("params__")}
            )
        else:
            add_job = st.sidebar.button("Add job")
            delete_all_jobs = st.sidebar.button("Delete all jobs")

            if add_job:
                if "jobs" not in st.session_state:
                    st.session_state.jobs = []
                st.session_state.jobs.append(
                    {k[len("params__") :]: v for k, v in st.session_state.items() if k.startswith("params__")}
                )

            if delete_all_jobs:
                st.session_state.jobs = []

    # show the grid with parameters for hub_model training
    selected_rows = []
    if "jobs" in st.session_state and "model_choice" in locals():
        if model_choice != "AutoTrain":
            if len(st.session_state.jobs) == 1 and "num_models" in st.session_state.jobs[0]:
                st.session_state.jobs = []
            if len(st.session_state.jobs) > 0:
                ag_resp = create_grid(st.session_state.jobs)
                ag_resp_sel = ag_resp["selected_rows"]
                if ag_resp_sel:
                    selected_rows = [
                        int(ag_resp_sel[i]["_selectedRowNodeInfo"]["nodeId"]) for i in range(len(ag_resp_sel))
                    ]
                    logger.info(selected_rows)
                st.markdown("<p>Only selected jobs will be used for training.</p>", unsafe_allow_html=True)

    # create project
    # step1: process dataset
    # step2: create project using AutoTrain API
    # step3: estimate costs
    # step4: start training if user confirms
    if not verify_project_name(project_name=project_name, username=autotrain_username, user_token=user_token):
        return
    if model_choice != "AutoTrain":
        if "jobs" not in st.session_state:
            st.error("Please add at least one job")
            return
        if len(st.session_state.jobs) == 0:
            st.error("Please add at least one job")
            return
        if len(selected_rows) == 0:
            st.error("Please select at least one job")
            return
    logger.info(st.session_state)

    # estimated_cost_button = st.button("Estimate Cost")
    # dset_available = False
    try:
        if task == "dreambooth":
            dset = AutoTrainDreamboothDataset(
                num_concepts=number_of_concepts,
                concept_images=[
                    st.session_state[f"dreambooth_concept_images_{i + 1}"] for i in range(number_of_concepts)
                ],
                concept_names=[
                    st.session_state[f"dreambooth_concept_name_{i + 1}"] for i in range(number_of_concepts)
                ],
                token=user_token,
                project_name=project_name,
                username=autotrain_username,
            )
        elif task.startswith("image"):
            dset = AutoTrainImageClassificationDataset(
                train_data=training_images,
                token=user_token,
                project_name=project_name,
                username=autotrain_username,
                valid_data=validation_images,
                percent_valid=None,  # TODO: add to UI
            )
        else:
            dset = AutoTrainDataset(
                train_data=training_data,
                task=task,
                token=user_token,
                project_name=project_name,
                username=autotrain_username,
                column_mapping={map_name: st.session_state[f"map_{map_name}"] for map_name in COLUMN_MAPPING[task]},
                valid_data=validation_data,
                percent_valid=None,  # TODO: add to UI
            )

        logger.info(f"Number of samples: {dset.num_samples}")
        estimated_cost = get_project_cost(
            username=autotrain_username,
            token=user_token,
            task=task,
            num_samples=dset.num_samples,
            num_models=len(selected_rows) if model_choice != "AutoTrain" else st.session_state.jobs[0]["num_models"],
        )
        st.info(f"Estimated cost: {estimated_cost} USD")
        # dset_available = True
    except Exception as e:
        logger.error(e)
        st.error("Error estimating costs. Please check your inputs and try again.")
        return

    # create project button
    create_project_button = st.button("Create Project")

    if create_project_button:
        # if not dset_available:
        #     st.error("Please estimate cost first.")

        with st.spinner("Munging data and uploading to 🤗 Hub..."):
            dset.prepare()

        project = Project(
            dataset=dset,
            hub_model=hub_model,
            job_params=get_job_params(st.session_state.jobs, selected_rows, task, model_choice),
        )
        with st.spinner("Creating project..."):
            project_id = project.create()
        with st.spinner("Approving project for training..."):
            project.approve(project_id)

        st.success(
            "Project created successfully. Monitor progess on the [dashboard](https://ui.autotrain.huggingface.co/projects)."
        )


if __name__ == "__main__":
    args = parse_args()
    if args.task is not None:
        st.session_state.task = args.task
    app()
