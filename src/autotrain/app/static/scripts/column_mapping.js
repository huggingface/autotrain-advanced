window.addEventListener("load", function () {
    function createColumnMappings(selectedTask) {
        const colMapDiv = document.getElementById("col_map_div");
        colMapDiv.innerHTML = ''; // Clear previous mappings

        let fields = [];

        switch (selectedTask) {
            case 'llm:sft':
            case 'llm:generic':
                fields = ['text'];
                break;
            case 'llm:dpo':
            case 'llm:orpo':
                fields = ['prompt', 'text', 'rejected_text'];
                break;
            case 'llm:reward':
                fields = ['text', 'rejected_text'];
                break;
            case 'text-classification':
            case 'seq2seq':
            case 'text-regression':
                fields = ['text', 'label'];
                break;
            case 'token-classification':
                fields = ['tokens', 'tags'];
                break;
            case 'dreambooth':
                fields = ['image'];
                break;
            case 'image-classification':
                fields = ['image', 'label'];
                break;
            case 'image-object-detection':
                fields = ['image', 'objects'];
                break;
            case 'tabular:classification':
            case 'tabular:regression':
                fields = ['id', 'label'];
                break;
            default:
                return; // Do nothing if task is not recognized
        }

        fields.forEach(field => {
            const fieldDiv = document.createElement('div');
            fieldDiv.className = 'mb-2';
            fieldDiv.innerHTML = `
  <label class="block text-gray-600 text-sm font-bold mb-1" for="col_map_${field}">
    ${field}:
  </label>
  <input type="text" id="col_map_${field}" name="col_map_${field}" value="${field}" class="mt-1 block w-full border border-gray-300 px-3 py-2 bg-white rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
`;
            colMapDiv.appendChild(fieldDiv);
        });
    }

    document.querySelector('select#task').addEventListener('change', (event) => {
        const selectedTask = event.target.value;
        createColumnMappings(selectedTask);
    });
    createColumnMappings(document.querySelector('select#task').value);
});