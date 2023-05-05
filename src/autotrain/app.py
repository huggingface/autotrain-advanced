import argparse
import os
import random
import re
import string

import pandas as pd
import streamlit as st
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from st_aggrid import AgGrid, AgGridTheme, ColumnsAutoSizeMode, GridOptionsBuilder, GridUpdateMode

from autotrain import help
from autotrain.dataset import AutoTrainDataset, AutoTrainDreamboothDataset, AutoTrainImageClassificationDataset
from autotrain.params import Params
from autotrain.project import Project
from autotrain.tasks import COLUMN_MAPPING
from autotrain.utils import app_error_handler, get_project_cost, get_user_token, user_authentication


APP_TASKS = {
    "Natural Language Processing": ["Text Classification", "LLM Finetuning"],
    # "Tabular": TABULAR_TASKS,
    "Computer Vision": ["Image Classification", "Dreambooth"],
}

APP_TASKS_MAPPING = {
    "Text Classification": "text_multi_class_classification",
    "LLM Finetuning": "lm_training",
    "Image Classification": "image_multi_class_classification",
    "Dreambooth": "dreambooth",
}

APP_TASK_TYPE_MAPPING = {
    "text_classification": "Natural Language Processing",
    "lm_training": "Natural Language Processing",
    "image_classification": "Computer Vision",
    "dreambooth": "Computer Vision",
}


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
    data_repo_name = f"{username}/autotrain-data-{project_name}"
    if does_repo_exist(data_repo_name, "dataset"):
        st.error("A project with this name already exists")
        return False
    return True


def get_job_params(job_params, selected_rows, task, param_choice):
    """
    Get job parameters list of dicts for AutoTrain and HuggingFace Hub models
    :param job_params: job parameters
    :param selected_rows: selected rows
    :param task: task
    :param param_choice: model choice
    :return: job parameters list of dicts
    """
    if param_choice == "AutoTrain":
        if len(job_params) > 1:
            raise ValueError("❌ Only one job parameter is allowed for AutoTrain.")
        job_params[0].update({"task": task})
    elif param_choice.lower() == "manual":
        for i in range(len(job_params)):
            job_params[i].update({"task": task})
        job_params = [job_params[i] for i in selected_rows]
    return job_params


def on_change_reset_jobs():
    # check if "jobs" exists in session_state
    if "jobs" in st.session_state:
        len_jobs = [len(j) for j in st.session_state.jobs]
        # if all lengths are not same, reset jobs
        if len(set(len_jobs)) != 1:
            st.session_state.jobs = []


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


@app_error_handler
def app():  # username, valid_orgs):
    st.sidebar.markdown(
        "<p style='text-align: center; font-size: 20px; font-weight: bold;'>AutoTrain Advanced</p>",
        unsafe_allow_html=True,
    )
    user_token = os.environ.get("HF_TOKEN", "")
    if len(user_token) == 0:
        user_token = get_user_token()
    if user_token is None:
        st.markdown(
            """Please login with a write [token](https://huggingface.co/settings/tokens).
            You can also pass your HF token in an environment variable called `HF_TOKEN` to avoid having to enter it every time.
            """
        )
        user_token = st.text_input("HuggingFace Token", type="password")

    if user_token is None:
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

    valid_can_pay = [username] + valid_orgs if user_can_pay else valid_orgs
    who_is_training = [username] + [org["name"] for org in orgs]

    st.markdown("###### Project Info")
    col1, col2 = st.columns(2)
    with col1:
        autotrain_username = st.selectbox("Who is training?", who_is_training, help=help.APP_AUTOTRAIN_USERNAME)
        can_pay = autotrain_username in valid_can_pay
    with col2:
        project_name = st.text_input("Project name", st.session_state.random_project_name, help=help.APP_PROJECT_NAME)

    if "task" in st.session_state:
        project_type = APP_TASK_TYPE_MAPPING[st.session_state.task]
        task = st.session_state.task
    else:
        col1, col2 = st.columns(2)
        with col1:
            project_type = st.selectbox("Project Type", list(APP_TASKS.keys()))
        with col2:
            if project_type == "Natural Language Processing":
                task = st.selectbox("Task", APP_TASKS[project_type])
            elif project_type == "Computer Vision":
                task = st.selectbox("Task", APP_TASKS[project_type])
            elif project_type == "Tabular":
                task = st.selectbox("Task", APP_TASKS[project_type])

    task = APP_TASKS_MAPPING[task]

    if task == "lm_training":
        lm_subtask = st.selectbox(
            "Subtask", ["Masked Language Modeling", "Causal Language Modeling"], index=1, disabled=True
        )
        if lm_subtask == "Causal Language Modeling":
            lm_training_type = st.selectbox(
                "Training Type",
                ["Generic", "Chat"],
                index=0,
                key="lm_training_type_choice",
                help=help.APP_LM_TRAINING_TYPE,
            )

    st.markdown("###### Model choice")
    if task.startswith("tabular"):
        model_choice = "AutoTrain"
    elif task == "lm_training":
        model_choice = "HuggingFace Hub"
    else:
        model_choice_label = ["AutoTrain", "HuggingFace Hub"]
        model_choice = st.selectbox(
            "Model Choice",
            model_choice_label,
            label_visibility="collapsed",
            on_change=on_change_reset_jobs(),
        )

    hub_model = None
    if model_choice == "HuggingFace Hub":
        default_hub_model = "bert-base-uncased"
        if task == "dreambooth":
            default_hub_model = "stabilityai/stable-diffusion-2-1-base"
        if task == "lm_training":
            default_hub_model = "EleutherAI/pythia-70m"
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
    else:
        tab1, tab2 = st.tabs(["Training", "Validation (Optional)"])
        with tab1:
            if project_type == "Computer Vision":
                training_images = st.file_uploader(
                    "Training Images", type=["zip"], help=help.APP_IMAGE_CLASSIFICATION_DATA_HELP
                )
            else:
                training_data = st.file_uploader("Training Data", type=["csv", "jsonl"], accept_multiple_files=True)

        with tab2:
            if project_type == "Computer Vision":
                validation_images = st.file_uploader(
                    "Validation Images", type=["zip"], help=help.APP_IMAGE_CLASSIFICATION_DATA_HELP
                )
            else:
                validation_data = st.file_uploader(
                    "Validation Data", type=["csv", "jsonl"], accept_multiple_files=True
                )

    if task not in ("dreambooth", "image_multi_class_classification"):
        if not ("training_data" in locals() and training_data):
            raise ValueError("Training data not found")
        st.markdown("###### Column mapping")
        # read column names
        # uploaded_file.seek(0)
        # temp_train_data = copy.deepcopy(training_data[0])
        if training_data[0].name.endswith(".csv"):
            df = pd.read_csv(training_data[0], nrows=0)
        elif training_data[0].name.endswith(".jsonl"):
            df = pd.read_json(training_data[0], lines=True, nrows=0)
        else:
            raise ValueError("Unknown file type")
        training_data[0].seek(0)
        # del temp_train_data
        columns = list(df.columns)
        if task == "lm_training":
            if lm_training_type == "Chat":
                col_mapping_options = st.multiselect(
                    "Which columns do you have in your data?",
                    ["Prompt", "Response", "Context", "Prompt Start"],
                    ["Prompt", "Response"],
                )
                st.selectbox("Map `prompt` to:", columns, key="map_prompt")
                st.selectbox("Map `response` to:", columns, key="map_response")

                if "Prompt Start" in col_mapping_options:
                    st.selectbox("Map `prompt_start` to:", columns, key="map_prompt_start")
                else:
                    st.session_state["map_prompt_start"] = None

                if "Context" in col_mapping_options:
                    st.selectbox("Map `context` to:", columns, key="map_context")
                else:
                    st.session_state["map_context"] = None

                st.session_state["map_text"] = None
            else:
                st.selectbox("Map `text` to:", columns, key="map_text")
                st.session_state["map_prompt"] = None
                st.session_state["map_context"] = None
                st.session_state["map_response"] = None
                st.session_state["map_prompt_start"] = None
        else:
            for map_idx, map_name in enumerate(COLUMN_MAPPING[task]):
                if map_name == "id" and task.startswith("tabular"):
                    st.selectbox(f"Map `{map_name}` to:", columns + [""], index=map_idx, key=f"map_{map_name}")
                else:
                    st.selectbox(f"Map `{map_name}` to:", columns, index=map_idx, key=f"map_{map_name}")

    st.sidebar.markdown("### Parameters")
    if model_choice != "AutoTrain":
        # on_change reset st.session_stat["jobs"]
        param_choice = st.sidebar.selectbox(
            "Parameter Choice",
            ["AutoTrain", "Manual"],
            key="param_choice",
            on_change=on_change_reset_jobs(),
        )
    else:
        param_choice = st.sidebar.selectbox(
            "Parameter Choice", ["AutoTrain", "Manual"], key="param_choice", index=0, disabled=True
        )
        st.sidebar.markdown("Hyperparameters are selected automagically for AutoTrain models")
    params = Params(
        task=task,
        param_choice="autotrain" if param_choice == "AutoTrain" else "manual",
        model_choice="autotrain" if model_choice == "AutoTrain" else "hub_model",
    )
    params = params.get()

    for key, value in params.items():
        if value.STREAMLIT_INPUT == "selectbox":
            if value.PRETTY_NAME == "LM Training Type":
                _choice = [lm_training_type.lower()]
                st.sidebar.selectbox(value.PRETTY_NAME, _choice, 0, key=f"params__{key}", disabled=True)
            else:
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
    if param_choice == "AutoTrain":
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
    if "jobs" in st.session_state and "param_choice" in locals():
        if param_choice != "AutoTrain":
            if len(st.session_state.jobs) == 1 and "num_models" in st.session_state.jobs[0]:
                st.session_state.jobs = []
            if len(st.session_state.jobs) > 0:
                ag_resp = create_grid(st.session_state.jobs)
                ag_resp_sel = ag_resp["selected_rows"]
                if ag_resp_sel:
                    selected_rows = [
                        int(ag_resp_sel[i]["_selectedRowNodeInfo"]["nodeId"]) for i in range(len(ag_resp_sel))
                    ]
                st.markdown("<p>Only selected jobs will be used for training.</p>", unsafe_allow_html=True)

    # create project
    # step1: process dataset
    # step2: create project using AutoTrain API
    # step3: estimate costs
    # step4: start training if user confirms
    if not verify_project_name(project_name=project_name, username=autotrain_username, user_token=user_token):
        return
    if param_choice != "AutoTrain":
        if "jobs" not in st.session_state:
            st.error("Please add at least one job")
            return
        if len(st.session_state.jobs) == 0:
            st.error("Please add at least one job")
            return
        if len(selected_rows) == 0:
            st.error("Please select at least one job")
            return

    if task == "dreambooth":
        concept_images = [
            st.session_state.get(f"dreambooth_concept_images_{i + 1}") for i in range(number_of_concepts)
        ]
        if sum(len(x) for x in concept_images) == 0:
            raise ValueError("Please upload concept images")
        dset = AutoTrainDreamboothDataset(
            num_concepts=number_of_concepts,
            concept_images=[st.session_state[f"dreambooth_concept_images_{i + 1}"] for i in range(number_of_concepts)],
            concept_names=[st.session_state[f"dreambooth_concept_name_{i + 1}"] for i in range(number_of_concepts)],
            token=user_token,
            project_name=project_name,
            username=autotrain_username,
        )
    elif task.startswith("image"):
        if not ("training_images" in locals() and training_images):
            raise ValueError("Please upload training images")
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

    estimated_cost = get_project_cost(
        username=autotrain_username,
        token=user_token,
        task=task,
        num_samples=dset.num_samples,
        num_models=len(selected_rows) if param_choice != "AutoTrain" else st.session_state.jobs[0]["num_models"],
    )
    st.info(f"Estimated cost: {estimated_cost} USD")
    if estimated_cost > 0 and can_pay is False:
        st.error(
            "You do not have enough credits to train this project. Please choose a user/org with a valid payment method attached to their account."
        )
        return

    if estimated_cost > 0 and can_pay is True:
        st.warning("Please note: clicking the create project button will start training and incur a cost!")

    # create project
    create_project_button = st.button(
        "Create Project" if estimated_cost == 0 else f"Create Project ({estimated_cost} USD)"
    )

    if create_project_button:
        with st.spinner("Munging data and uploading to 🤗 Hub..."):
            dset.prepare()

        project = Project(
            dataset=dset,
            param_choice=param_choice,
            hub_model=hub_model,
            job_params=get_job_params(st.session_state.jobs, selected_rows, task, param_choice),
        )
        with st.spinner("Creating project..."):
            project_id = project.create()
        with st.spinner("Approving project for training..."):
            project.approve(project_id)

        st.success(
            f"Project created successfully. Monitor progess on the [dashboard](https://ui.autotrain.huggingface.co/{project_id}/trainings)."
        )


if __name__ == "__main__":
    args = parse_args()
    if args.task is not None:
        st.session_state.task = args.task
    # generate a random project name separated by hyphens, e.g. 43vs-3sd3-2355
    if "random_project_name" not in st.session_state:
        st.session_state.random_project_name = "-".join(
            ["".join(random.choices(string.ascii_lowercase + string.digits, k=4)) for _ in range(3)]
        )
    app()
