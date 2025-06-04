document.addEventListener('DOMContentLoaded', function () {
    function fetchDataAndUpdateModels() {
        const taskValue = document.getElementById('task').value;
        const baseModelSelect = document.getElementById('base_model');
        const queryParams = new URLSearchParams(window.location.search);
        const customModelsValue = queryParams.get('custom_models');
        const baseModelInput = document.getElementById('base_model_input');
        const baseModelCheckbox = document.getElementById('base_model_checkbox');

        let fetchURL = `/ui/model_choices/${taskValue}`;
        if (customModelsValue) {
            fetchURL += `?custom_models=${customModelsValue}`;
        }
        baseModelSelect.innerHTML = 'Fetching models...';
        fetch(fetchURL)
            .then(response => response.json())
            .then(data => {
                const baseModelSelect = document.getElementById('base_model');
                baseModelCheckbox.checked = false;
                baseModelSelect.classList.remove('hidden');
                baseModelInput.classList.add('hidden');
                baseModelSelect.innerHTML = ''; // Clear existing options
                data.forEach(model => {
                    let option = document.createElement('option');
                    option.value = model.id; // Assuming each model has an 'id'
                    option.textContent = model.name; // Assuming each model has a 'name'
                    baseModelSelect.appendChild(option);
                });
            })
            .catch(error => console.error('Error:', error));
    }
    document.getElementById('task').addEventListener('change', fetchDataAndUpdateModels);
    fetchDataAndUpdateModels();
});