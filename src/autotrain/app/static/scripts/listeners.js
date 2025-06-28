document.addEventListener('DOMContentLoaded', function () {
    const dataSource = document.getElementById("dataset_source");
    const uploadDataTabContent = document.getElementById("upload-data-tab-content");
    const hubDataTabContent = document.getElementById("hub-data-tab-content");
    const uploadDataTabs = document.getElementById("upload-data-tabs");

    const jsonCheckbox = document.getElementById('show-json-parameters');
    const jsonParametersDiv = document.getElementById('json-parameters');
    const dynamicUiDiv = document.getElementById('dynamic-ui');

    const paramsTextarea = document.getElementById('params_json');

    const updateTextarea = () => {
        const paramElements = document.querySelectorAll('[id^="param_"]');
        const params = {};
        paramElements.forEach(el => {
            const key = el.id.replace('param_', '');
            params[key] = el.value;
        });
        paramsTextarea.value = JSON.stringify(params, null, 2);
        paramsTextarea.style.height = '600px';
    };

    const observeParamChanges = () => {
        const paramElements = document.querySelectorAll('[id^="param_"]');
        paramElements.forEach(el => {
            el.addEventListener('input', updateTextarea);
        });
    };

    const updateParamsFromTextarea = () => {
        try {
            const params = JSON.parse(paramsTextarea.value);
            Object.keys(params).forEach(key => {
                const el = document.getElementById('param_' + key);
                if (el) {
                    el.value = params[key];
                }
            });
        } catch (e) {
            console.error('Invalid JSON:', e);
        }
    };

    function switchToJSON() {
        if (jsonCheckbox.checked) {
            dynamicUiDiv.style.display = 'none';
            jsonParametersDiv.style.display = 'block';
        } else {
            dynamicUiDiv.style.display = 'block';
            jsonParametersDiv.style.display = 'none';
        }
    }

    // Function to handle changes in dataset_source and task
    function handleDataSource() {
        const lifeAppSelection = document.getElementById("life-app-selection");
        const datasetFileDiv = document.getElementById('dataset_file_div');
        const taskValue = document.getElementById('task').value;

        // Hide all data source related sections by default
        if (hubDataTabContent) hubDataTabContent.style.display = "none";
        if (uploadDataTabContent) uploadDataTabContent.style.display = "none";
        if (uploadDataTabs) uploadDataTabs.style.display = "none";
        if (lifeAppSelection) lifeAppSelection.style.display = "none";
        if (datasetFileDiv) datasetFileDiv.style.display = 'none';

        // Show/hide LiFE App option based on task
        const lifeAppOption = document.getElementById("dataset_source").querySelector('option[value="life_app"]');
        if (lifeAppOption) {
            if (taskValue === "ASR") {
                lifeAppOption.style.display = ""; 
            } else {
                lifeAppOption.style.display = "none"; 
                
        if (dataSource.value === "life_app") {
                    dataSource.value = "local";
                }
            }
        }

        // Show relevant section based on selected data source
        if (dataSource.value === "life_app" && taskValue === "ASR") {
            if (lifeAppSelection) lifeAppSelection.style.display = "block";
            if (datasetFileDiv) datasetFileDiv.style.display = 'block';
            loadLifeAppProjects();
            // loadLifeAppScripts();
            // loadDatasetFiles();
        } else if (dataSource.value === "huggingface") {
            if (hubDataTabContent) hubDataTabContent.style.display = "block";
        } else if (dataSource.value === "local") {
            if (uploadDataTabContent) uploadDataTabContent.style.display = "block";
            if (uploadDataTabs) uploadDataTabs.style.display = "block";
        }
    }

    async function fetchParams() {
        const taskValue = document.getElementById('task').value;
        const parameterMode = document.getElementById('parameter_mode').value;
        const response = await fetch(`/ui/params/${taskValue}/${parameterMode}`);
        const params = await response.json();
        return params;
    }

    function createElement(param, config) {
        let element = '';
        switch (config.type) {
            case 'number':
                element = `<div>
                    <label for="param_${param}" class="text-sm font-medium text-gray-700 dark:text-gray-300">${config.label}</label>
                    <input type="number" name="param_${param}" id="param_${param}" value="${config.default}"
                        class="mt-1 p-1 text-xs font-medium w-full border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                </div>`;
                break;
            case 'dropdown':
                let options = config.options.map(option => `<option value="${option}" ${option === config.default ? 'selected' : ''}>${option}</option>`).join('');
                element = `<div>
                    <label for="param_${param}" class="text-sm font-medium text-gray-700 dark:text-gray-300">${config.label}</label>
                    <select name="param_${param}" id="param_${param}"
                        class="mt-1 p-1 text-xs font-medium w-full border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                        ${options}
                    </select>
                </div>`;
                break;
            case 'checkbox':
                element = `<div>
                    <label for="param_${param}" class="text-sm font-medium text-gray-700 dark:text-gray-300">${config.label}</label>
                    <input type="checkbox" name="param_${param}" id="param_${param}" ${config.default ? 'checked' : ''}
                        class="mt-1 text-xs font-medium border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                </div>`;
                break;
            case 'string':
                element = `<div>
                    <label for="param_${param}" class="text-sm font-medium text-gray-700 dark:text-gray-300">${config.label}</label>
                    <input type="text" name="param_${param}" id="param_${param}" value="${config.default}"
                        class="mt-1 p-1 text-xs font-medium w-full border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                </div>`;
                break;
        }
        return element;
    }

    function renderUI(params) {
        const uiContainer = document.getElementById('dynamic-ui');
        let rowDiv = null;
        let rowIndex = 0;
        let lastType = null;

        Object.keys(params).forEach((param, index) => {
            const config = params[param];
            if (lastType !== config.type || rowIndex >= 3) {
                if (rowDiv) uiContainer.appendChild(rowDiv);
                rowDiv = document.createElement('div');
                rowDiv.className = 'grid grid-cols-3 gap-2 mb-2';
                rowIndex = 0;
            }
            rowDiv.innerHTML += createElement(param, config);
            rowIndex++;
            lastType = config.type;
        });
        if (rowDiv) uiContainer.appendChild(rowDiv);
    }

    fetchParams().then(params => renderUI(params));

    document.getElementById('task').addEventListener('change', function () {
        fetchParams().then(params => {
            document.getElementById('dynamic-ui').innerHTML = '';
            let jsonCheckboxFlag = false;
            if (jsonCheckbox.checked) {
                jsonCheckbox.checked = false;
                jsonCheckboxFlag = true;
            }
            renderUI(params);
            if (jsonCheckboxFlag) {
                jsonCheckbox.checked = true;
                updateTextarea();
                observeParamChanges();
            }
            handleDataSource(); 
        });
    });

    document.getElementById('parameter_mode').addEventListener('change', function () {
        fetchParams().then(params => {
            document.getElementById('dynamic-ui').innerHTML = '';
            let jsonCheckboxFlag = false;
            if (jsonCheckbox.checked) {
                jsonCheckbox.checked = false;
                jsonCheckboxFlag = true;
            }
            renderUI(params);
            if (jsonCheckboxFlag) {
                jsonCheckbox.checked = true;
                updateTextarea();
                observeParamChanges();
            }
        });
    });

    jsonCheckbox.addEventListener('change', function () {
        if (jsonCheckbox.checked) {
            updateTextarea();
            observeParamChanges();
        }
    });

    document.getElementById('task').addEventListener('change', function () {
        if (jsonCheckbox.checked) {
            updateTextarea();
            observeParamChanges();
        }
    });

    // Attach event listeners to dataset_source dropdown and task dropdown
    dataSource.addEventListener("change", handleDataSource);
    document.getElementById('task').addEventListener('change', handleDataSource); 
    jsonCheckbox.addEventListener('change', switchToJSON);
    paramsTextarea.addEventListener('input', updateParamsFromTextarea);

    // Trigger the event listener to set the initial state
    handleDataSource();
    observeParamChanges();
    updateTextarea();

    // LiFE App Dataset Selection (API vs JSON)
    document.getElementById('life-app-source').addEventListener('change', function() {
        const apiContainer = document.getElementById('life-app-api-container');
        const jsonContainer = document.getElementById('life-app-json-container');
        
        if (this.value === 'api') {
            apiContainer.style.display = 'block';
            jsonContainer.style.display = 'none';
        } else if (this.value === 'json') {
            apiContainer.style.display = 'none';
            jsonContainer.style.display = 'block';
        } else {
            apiContainer.style.display = 'none';
            jsonContainer.style.display = 'none';
        }
    });

    // Handle JSON file selection
    document.getElementById('life-app-json-file').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const data = JSON.parse(e.target.result);
                    if (!Array.isArray(data)) {
                        throw new Error('JSON must be an array');
                    }
                    if (data.length === 0) {
                        throw new Error('JSON array is empty');
                    }
                    const firstItem = data[0];
                    if (!firstItem.audio || !firstItem.transcription) {
                        throw new Error('Each item must have audio and transcription fields');
                    }
                    window.lifeAppData = data;
                } catch (error) {
                    alert('Invalid JSON file: ' + error.message);
                    e.target.value = '';
                }
            };
            reader.readAsText(file);
        }
    });

    // Function to load projects
    async function loadLifeAppProjects() {
        const projectSelect = document.getElementById('life_app_project');
        if (!projectSelect) return;
        
        try {
            // Fetch from backend API
            const response = await fetch('/ui/life_app_projects');
            if (!response.ok) {
                throw new Error('Failed to fetch projects');
        }
            const data = await response.json();
            const projects = data.projects || [];
            
            // Clear existing options
            projectSelect.innerHTML = '';
            
            // Add options to select
            projects.forEach(project => {
                const option = document.createElement('option');
                option.value = project;
                option.textContent = project;
                projectSelect.appendChild(option);
            });
            
            // Initialize Select2 if not already initialized
            if (!$(projectSelect).data('select2')) {
                $(projectSelect).select2({
                    placeholder: "Select LiFE App Project(s)",
                    allowClear: true,
                    multiple: true,
                    width: '100%',
                    maximumSelectionLength: 2,
                    tags: true
                });
            }
            
            // Update tags display
            updateProjectTags();
            
            // Add change event listener for Select2
            $(projectSelect).off('change').on('change', function() {
                console.log('[DYNAMIC HANDLER] #life_app_project changed:', $(this).val());
                updateProjectTags();
                const selectedProjects = $(this).val();
                if (selectedProjects && selectedProjects.length > 0) {
                    console.log('[DYNAMIC HANDLER] Calling loadScriptsForProjects with:', selectedProjects);
                    loadScriptsForProjects(selectedProjects);
                } else {
                    $('#life_app_script').prop('disabled', true).empty();
                }
                });
        } catch (error) {
            console.error('Error loading projects:', error);
        }
    }

    // Initialize Select2 for script dropdown
    $('#life_app_script').select2({
        placeholder: "Select Script",
        allowClear: true,
        width: '100%'
    });

    // Function to load scripts for selected projects
    async function loadScriptsForProjects(selectedProjects) {
        const scriptSelect = $('#life_app_script');
        scriptSelect.prop('disabled', false).empty();
        scriptSelect.append(new Option('Select Script', ''));

        try {
            const response = await fetch('/ui/project_selected', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ projects: selectedProjects })
            });
            if (!response.ok) throw new Error('Failed to fetch scripts');
            const data = await response.json();
            if (data.scripts) {
                data.scripts.forEach(script => {
                    scriptSelect.append(new Option(script, script));
                });
            }

            // Reinitialize Select2 after adding options
            if (scriptSelect.data('select2')) {
                scriptSelect.select2('destroy');
            }
            scriptSelect.select2({
                placeholder: "Select Script",
                allowClear: true,
                width: '100%'
            });

            // Restore previous selection if possible and force UI update
            if (window.selectedScript && data.scripts.includes(window.selectedScript)) {
                scriptSelect.val(window.selectedScript).trigger('change');
            } else if (data.scripts && data.scripts.length === 1) {
                scriptSelect.val(data.scripts[0]).trigger('change');
            } else {
                scriptSelect.val('').trigger('change');
            }

            // Always re-attach the change handler after Select2 re-init
            scriptSelect.off('change').on('change', function() {
            const selectedScript = $(this).val();
                window.selectedScript = selectedScript;
            if (selectedScript) {
                    loadDatasetsForScript(selectedScript);
                } else {
                    $('#dataset_file').prop('disabled', true).empty();
                }
            });
        } catch (error) {
            console.error('Error loading scripts:', error);
        }
    }

    // Project selection event
    $('#life_app_project').on('change', function() {
        console.log('[STATIC HANDLER] #life_app_project changed:', $(this).val());
        const selectedProjects = $(this).val();
        if (selectedProjects && selectedProjects.length > 0) {
            console.log('[STATIC HANDLER] Calling loadScriptsForProjects with:', selectedProjects);
            loadScriptsForProjects(selectedProjects);
            } else {
            $('#life_app_script').prop('disabled', true).empty();
            $('#dataset_file').prop('disabled', true).empty();
        }
    });

    async function loadDatasetsForScript(selectedScript) {
        console.log('[FRONTEND] Sending selected script to backend:', selectedScript);
        const datasetSelect = $('#dataset_file');
        datasetSelect.prop('disabled', false).empty();
        datasetSelect.append(new Option('Select Dataset', ''));
    
        try {
            const response = await fetch('/ui/script_selected', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    script: selectedScript 
                })
            });
    
            if (!response.ok) {
                throw new Error('Failed to fetch datasets');
            }
    
            const data = await response.json();
            console.log('[FRONTEND] Datasets received from backend for script', selectedScript, ':', data.datasets);
            if (data.datasets && data.datasets.length > 0) {
                data.datasets.forEach(dataset => {
                    datasetSelect.append(new Option(dataset, dataset));
                });
            }
    
            // Reinitialize Select2
            if (datasetSelect.data('select2')) {
                datasetSelect.select2('destroy');
            }
            datasetSelect.select2({
                placeholder: "Select Dataset File",
                allowClear: true,
                width: '100%'
            });
    
            datasetSelect.trigger('change');
        } catch (error) {
            console.error('Error loading datasets:', error);
        }
    }

    // Function to update project tags
    function updateProjectTags() {
        const projectSelect = document.getElementById('life_app_project');
        const tagContainer = document.getElementById('life-app-project-tags');
        if (!projectSelect || !tagContainer) return;
        
        // Clear existing tags
        tagContainer.innerHTML = '';
        
        // Get selected options
        const selectedOptions = $(projectSelect).select2('data');
        
        // Create tags for each selected option
        selectedOptions.forEach(option => {
            const tag = document.createElement('span');
            tag.className = 'inline-flex items-center px-2 py-1 mr-2 mb-2 text-sm font-medium text-blue-800 bg-blue-100 rounded-full dark:bg-blue-900 dark:text-blue-300';
            tag.textContent = option.text;
            
            const removeButton = document.createElement('button');
            removeButton.className = 'ml-1 text-blue-800 hover:text-blue-600 dark:text-blue-300 dark:hover:text-blue-100';
            removeButton.innerHTML = '×';
            removeButton.onclick = () => {
                const optionElement = Array.from(projectSelect.options).find(opt => opt.value === option.id);
                if (optionElement) {
                    optionElement.selected = false;
                    $(projectSelect).trigger('change');
            }
            };
            
            tag.appendChild(removeButton);
            tagContainer.appendChild(tag);
    });
    }

    $('#dataset_file').find('option').each(function() { console.log($(this).val()); });
});