document.addEventListener('DOMContentLoaded', function () {
    var fetchLogsInterval;

    // Function to check the modal's display property and fetch logs if visible
    function fetchAndDisplayLogs() {
        var modal = document.getElementById('logs-modal');
        var displayStyle = window.getComputedStyle(modal).display;

        // Check if the modal display property is 'flex'
        if (displayStyle === 'flex') {
            fetchLogs(); // Initial fetch when the modal is opened

            // Clear any existing interval to avoid duplicates
            clearInterval(fetchLogsInterval);

            // Set up the interval to fetch logs every 5 seconds
            fetchLogsInterval = setInterval(fetchLogs, 5000);
        } else {
            // Clear the interval when the modal is not displayed as 'flex'
            clearInterval(fetchLogsInterval);
        }
    }

    // Function to fetch logs from the server
    function fetchLogs() {
        fetch('/ui/logs')
            .then(response => response.json())
            .then(data => {
                var logContainer = document.getElementById('logContent');
                logContainer.innerHTML = ''; // Clear previous logs

                // Handling the case when logs are only available in local mode or no logs available
                if (typeof data.logs === 'string') {
                    logContainer.textContent = data.logs;
                } else {
                    // Assuming data.logs is an array of log entries
                    data.logs.forEach(log => {
                        if (log.trim().length > 0) {
                            var p = document.createElement('p');
                            p.textContent = log;
                            logContainer.appendChild(p); // Appends logs in order received
                        }
                    });
                }
            })
            .catch(error => console.error('Error fetching logs:', error));
    }

    // Set up an observer to detect when the modal becomes visible or hidden
    var observer = new MutationObserver(function (mutations) {
        mutations.forEach(function (mutation) {
            if (mutation.attributeName === 'class') {
                fetchAndDisplayLogs();
            }
        });
    });

    var modal = document.getElementById('logs-modal');
    observer.observe(modal, {
        attributes: true //configure it to listen to attribute changes
    });
});