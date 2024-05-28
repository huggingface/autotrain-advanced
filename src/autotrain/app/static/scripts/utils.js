document.addEventListener('DOMContentLoaded', function () {

    const loadingSpinner = document.getElementById('loadingSpinner');

    function generateRandomString(length) {
        let result = '';
        const characters = 'abcdefghijklmnopqrstuvwxyz0123456789';
        const charactersLength = characters.length;
        for (let i = 0; i < length; i++) {
            result += characters.charAt(Math.floor(Math.random() * charactersLength));
        }
        return result;
    }

    function setRandomProjectName() {
        const part1 = generateRandomString(5);
        const part2 = generateRandomString(5);
        const randomName = `autotrain-${part1}-${part2}`;
        document.getElementById('project_name').value = randomName;
    }

    function showFinalModal() {
        const modal = document.getElementById('final-modal');
        modal.classList.add('flex');
        modal.classList.remove('hidden');
    }

    function hideFinalModal() {
        const modal = document.getElementById('final-modal');
        modal.classList.remove('flex');
        modal.classList.add('hidden');
    }

    function showModal() {
        const modal = document.getElementById('confirmation-modal');
        modal.classList.add('flex');
        modal.classList.remove('hidden');
    }

    function showLogsModal() {
        const modal = document.getElementById('logs-modal');
        modal.classList.add('flex');
        modal.classList.remove('hidden');
    }

    function hideLogsModal() {
        const modal = document.getElementById('logs-modal');
        modal.classList.remove('flex');
        modal.classList.add('hidden');
    }

    function hideModal() {
        const modal = document.getElementById('confirmation-modal');
        modal.classList.remove('flex');
        modal.classList.add('hidden');
    }

    document.getElementById('start-training-button').addEventListener('click', function () {
        showModal();
    });

    document.querySelector('#confirmation-modal .confirm').addEventListener('click', async function () {
        hideModal();
        loadingSpinner.classList.remove('hidden');
        console.log(document.getElementById('params_json').value)

        var formData = new FormData();
        var columnMapping = {};
        var params;
        var paramsJsonElement = document.getElementById('params_json');
        document.querySelectorAll('[id^="col_map_"]').forEach(function (element) {
            var key = element.id.replace('col_map_', '');
            columnMapping[key] = element.value;
        });

        if (paramsJsonElement.value == '{}' || paramsJsonElement.value == '') {
            var paramsDict = {};
            document.querySelectorAll('[id^="param_"]').forEach(function (element) {
                var key = element.id.replace('param_', '');
                paramsDict[key] = element.value;
            });
            params = JSON.stringify(paramsDict);
        } else {
            params = paramsJsonElement.value;
        }
        formData.append('project_name', document.getElementById('project_name').value);
        formData.append('task', document.getElementById('task').value);
        formData.append('base_model', document.getElementById('base_model').value);
        formData.append('hardware', document.getElementById('hardware').value);
        formData.append('params', params);
        formData.append('autotrain_user', document.getElementById('autotrain_user').value);
        formData.append('column_mapping', JSON.stringify(columnMapping));
        formData.append('hub_dataset', document.getElementById('hub_dataset').value);
        formData.append('train_split', document.getElementById('train_split').value);
        formData.append('valid_split', document.getElementById('valid_split').value);

        var trainingFiles = document.getElementById('data_files_training').files;
        for (var i = 0; i < trainingFiles.length; i++) {
            formData.append('data_files_training', trainingFiles[i]);
        }

        var validationFiles = document.getElementById('data_files_valid').files;
        for (var i = 0; i < validationFiles.length; i++) {
            formData.append('data_files_valid', validationFiles[i]);
        }

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/ui/create_project', true);

        xhr.onload = function () {
            loadingSpinner.classList.add('hidden');
            var finalModalContent = document.querySelector('#final-modal .text-center');

            if (xhr.status === 200) {
                var responseObj = JSON.parse(xhr.responseText);
                var monitorURL = responseObj.monitor_url;
                if (monitorURL.startsWith('http')) {
                    finalModalContent.innerHTML = '<p>Success!</p>' +
                        '<p>You can check the progress of your training here: <a href="' + monitorURL + '" target="_blank">' + monitorURL + '</a></p>';
                } else {
                    finalModalContent.innerHTML = '<p>Success!</p>' +
                        '<p>' + monitorURL + '</p>';
                }

                showFinalModal();
            } else {
                finalModalContent.innerHTML = '<p>Error: ' + xhr.status + ' ' + xhr.statusText + '</p>' + '<p> Please check the logs for more information.</p>';
                console.error('Error:', xhr.status, xhr.statusText);
                showFinalModal();
            }
        };

        xhr.send(formData);
    });

    document.querySelector('#confirmation-modal .cancel').addEventListener('click', function () {
        hideModal();
    });

    document.querySelector('#final-modal button').addEventListener('click', function () {
        hideFinalModal();
    });

    document.querySelector('#button_logs').addEventListener('click', function () {
        showLogsModal();
    });

    document.querySelector('[data-modal-hide="logs-modal"]').addEventListener('click', function () {
        hideLogsModal();
    });
    document.getElementById('success-message').textContent = '';
    document.getElementById('error-message').textContent = '';

    document.getElementById('data_files_training').addEventListener('change', function () {
        var fileContainer = document.getElementById('file-container-training');
        var files = this.files;
        var fileText = '';

        for (var i = 0; i < files.length; i++) {
            fileText += files[i].name + ' ';
        }

        fileContainer.innerHTML = fileText;
    });
    document.getElementById('data_files_valid').addEventListener('change', function () {
        var fileContainer = document.getElementById('file-container-valid');
        var files = this.files;
        var fileText = '';

        for (var i = 0; i < files.length; i++) {
            fileText += files[i].name + ' ';
        }

        fileContainer.innerHTML = fileText;
    });

    window.onload = setRandomProjectName;
});
