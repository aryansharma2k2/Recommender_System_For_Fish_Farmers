// script.js
function generatePondFields() {
    var numPonds = document.getElementById('num_ponds').value;
    var pondFieldsContainer = document.getElementById('pond_fields_container');
    pondFieldsContainer.innerHTML = ''; // Clear previous fields

    for (var i = 1; i <= numPonds; i++) {
        var div = document.createElement('div');

        var labelLength = document.createElement('label');
        labelLength.innerHTML = 'Pond ' + i + ' - Length:';
        var inputLength = document.createElement('input');
        inputLength.type = 'number';
        inputLength.name = 'length_' + i;
        inputLength.step = 'any';
        inputLength.required = true;

        var labelBreadth = document.createElement('label');
        labelBreadth.innerHTML = 'Breadth:';
        var inputBreadth = document.createElement('input');
        inputBreadth.type = 'number';
        inputBreadth.name = 'breadth_' + i;
        inputBreadth.step = 'any';
        inputBreadth.required = true;

        var labelDepth = document.createElement('label');
        labelDepth.innerHTML = 'Depth:';
        var inputDepth = document.createElement('input');
        inputDepth.type = 'number';
        inputDepth.name = 'depth_' + i;
        inputDepth.step = 'any';
        inputDepth.required = true;

        div.appendChild(labelLength);
        div.appendChild(inputLength);
        div.appendChild(labelBreadth);
        div.appendChild(inputBreadth);
        div.appendChild(labelDepth);
        div.appendChild(inputDepth);

        pondFieldsContainer.appendChild(div);
    }
}
