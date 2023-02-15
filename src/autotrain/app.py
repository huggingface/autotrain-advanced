import copy
import re

import pandas as pd
import streamlit as st
from loguru import logger
from st_aggrid import AgGrid, AgGridTheme, ColumnsAutoSizeMode, GridOptionsBuilder, GridUpdateMode

from autotrain.dataset import Dataset
from autotrain.params import Params
from autotrain.project import Project
from autotrain.tasks import COLUMN_MAPPING, NLP_TASKS, TABULAR_TASKS, VISION_TASKS
from autotrain.utils import get_user_token, user_authentication


def verify_project_name(project_name, username):
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
    user_token = get_user_token()
    if user_token is None:
        st.error("You need to be logged in to create a project. Please login using `huggingface-cli login`")
        return False
    # data_repo_name = f"{username}/{project_name}"
    # TODO: make sure that data repo does not exist
    return True


def get_job_params(job_params, selected_rows, task, model_choice):
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


def app():  # username, valid_orgs):
    st.sidebar.markdown(
        "<p style='text-align: center; font-size: 20px; font-weight: bold;'>AutoTrain Advanced</p>",
        unsafe_allow_html=True,
    )
    user_token = get_user_token()
    if user_token is None:
        st.error("You need to be logged in to create a project. Please login using `huggingface-cli login`")
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
            "Please attach a CC to your account / join an organization with a CC attached to it to create a project"
        )
        return

    who_is_training = [username] + valid_orgs

    col1, col2 = st.columns(2)
    with col1:
        autotrain_username = st.selectbox("Who is training?", who_is_training)
    with col2:
        project_name = st.text_input("Project name", "my-project")

    col1, col2 = st.columns(2)
    with col1:
        project_type = st.selectbox(
            "Project Type",
            [
                "Natural Language Processing",
                "Computer Vision",
                "Tabular",
            ],
        )
    with col2:
        if project_type == "Natural Language Processing":
            task = st.selectbox("Task", list(NLP_TASKS.keys()))
        elif project_type == "Computer Vision":
            task = st.selectbox("Task", list(VISION_TASKS.keys()))
        elif project_type == "Tabular":
            task = st.selectbox("Task", list(TABULAR_TASKS.keys()))

    col1, col2 = st.columns(2)
    with col1:
        training_data = st.file_uploader("Training data", type=["csv", "jsonl"], accept_multiple_files=True)
    with col2:
        validation_data = st.file_uploader("Validation data", type=["csv", "jsonl"], accept_multiple_files=True)

    if training_data:
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
            st.selectbox(f"Map `{map_name}` to:", columns, index=map_idx, key=f"map_{map_name}")

        st.markdown("###### Model choice")
        model_choice = st.selectbox(
            "Model Choice",
            ["AutoTrain", "HuggingFace Hub"],
            label_visibility="collapsed",
        )
        hub_model = None
        if model_choice == "HuggingFace Hub":
            hub_model = st.text_input("Model name", "bert-base-uncased")

        st.sidebar.markdown("### Parameters")
        params = Params(task=task, training_type="autotrain" if model_choice == "AutoTrain" else "hub_model")
        params = params.get()

        if model_choice == "AutoTrain":
            st.sidebar.markdown("Hyperparameters are selected automagically for AutoTrain models")

        for key, value in params.items():
            if value.STREAMLIT_INPUT == "selectbox":
                st.sidebar.selectbox(value.PRETTY_NAME, value.CHOICES, 0, key=f"params__{key}")
            elif value.STREAMLIT_INPUT == "number_input":
                st.sidebar.number_input(
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

    selected_rows = []
    if "jobs" in st.session_state and model_choice != "AutoTrain":
        if len(st.session_state.jobs) == 1 and "num_models" in st.session_state.jobs[0]:
            st.session_state.jobs = []
        if len(st.session_state.jobs) > 0:
            df = pd.DataFrame(st.session_state.jobs)
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
            ag_resp_sel = ag_resp["selected_rows"]
            if ag_resp_sel:
                selected_rows = [
                    int(ag_resp_sel[i]["_selectedRowNodeInfo"]["nodeId"]) for i in range(len(ag_resp_sel))
                ]
                logger.info(selected_rows)
            st.markdown("<p>Only selected jobs will be used for training.</p>", unsafe_allow_html=True)

    create_project_button = st.button("Create Project")

    if create_project_button:
        if not verify_project_name(project_name=project_name, username=autotrain_username):
            return
        if model_choice != "AutoTrain":
            if "jobs" not in st.session_state:
                st.error("Please add at least one job")
                return
            if len(st.session_state.jobs) == 0:
                st.error("Please add at least one job")
                return
        logger.info(st.session_state)

        dset = Dataset(
            train_data=training_data,
            task=task,
            token=user_token,
            project_name=project_name,
            username=autotrain_username,
            column_mapping={map_name: st.session_state[f"map_{map_name}"] for map_name in COLUMN_MAPPING[task]},
            valid_data=validation_data,
            percent_valid=None,  # TODO: add to UI
        )
        with st.spinner("Munging data and uploading to 🤗 Hub..."):
            dset.prepare()

        project = Project(
            token=user_token,
            name=project_name,
            username=autotrain_username,
            task=task,
            hub_model=hub_model,
            job_params=get_job_params(st.session_state.jobs, selected_rows, task, model_choice),
        )
        project.create()


if __name__ == "__main__":
    app()
