document.addEventListener('DOMContentLoaded', (event) => {
    function pollAccelerators() {
        const numAcceleratorsElement = document.getElementById('num_accelerators');
        if (autotrain_local_value === 0) {
            numAcceleratorsElement.innerText = 'Accelerators: Only available in local mode.';
            numAcceleratorsElement.style.display = 'block'; // Ensure the element is visible
            return;
        }

        // Send a request to the /accelerators endpoint
        fetch('/ui/accelerators')
            .then(response => response.json()) // Assuming the response is in JSON format
            .then(data => {
                // Update the paragraph with the number of accelerators
                document.getElementById('num_accelerators').innerText = `Accelerators: ${data.accelerators}`;
            })
            .catch(error => {
                console.error('Error:', error);
                // Update the paragraph to show an error message
                document.getElementById('num_accelerators').innerText = 'Accelerators: Error fetching data';
            });
    }
    function pollModelTrainingStatus() {
        // Send a request to the /is_model_training endpoint

        if (autotrain_local_value === 0) {
            const statusParagraph = document.getElementById('is_model_training');
            statusParagraph.innerText = 'Running jobs: Only available in local mode.';
            statusParagraph.style.display = 'block';
            return;
        }
        fetch('/ui/is_model_training')
            .then(response => response.json()) // Assuming the response is in JSON format
            .then(data => {
                // Construct the message to display
                let message = data.model_training ? 'Running job PID(s): ' + data.pids.join(', ') : 'No running jobs';

                // Update the paragraph with the status of model training
                let statusParagraph = document.getElementById('is_model_training');
                statusParagraph.innerText = message;
                let stopTrainingButton = document.getElementById('stop-training-button');
                let startTrainingButton = document.getElementById('start-training-button');

                // Change the text color based on the model training status
                if (data.model_training) {
                    // Set text color to red if jobs are running
                    statusParagraph.style.color = 'red';
                    stopTrainingButton.style.display = 'block';
                    startTrainingButton.style.display = 'none';
                } else {
                    // Set text color to green if no jobs are running
                    statusParagraph.style.color = 'green';
                    stopTrainingButton.style.display = 'none';
                    startTrainingButton.style.display = 'block';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                // Update the paragraph to show an error message
                let statusParagraph = document.getElementById('is_model_training');
                statusParagraph.innerText = 'Error fetching training status';
                statusParagraph.style.color = 'red'; // Set error message color to red
            });
    }

    setInterval(pollAccelerators, 10000);
    setInterval(pollModelTrainingStatus, 5000);
    pollAccelerators();
    pollModelTrainingStatus();
});