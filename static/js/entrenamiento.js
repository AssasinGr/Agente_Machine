document.getElementById('trainingForm').addEventListener('submit', function(event) {
    event.preventDefault(); 

    document.getElementById('loadingSpinner').style.display = 'block'; 
    document.getElementById('results').style.display = 'none'; 

    const formData = new FormData(this);
    const data = {};
    formData.forEach((value, key) => { data[key] = value; });

    fetch('/entrenamiento', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loadingSpinner').style.display = 'none'; 
        document.getElementById('results').style.display = 'block'; 

        document.getElementById('episodios_entrenados').textContent = data.episodios_entrenados;
        document.getElementById('recompensa_promedio').textContent = data.recompensa_promedio_ultimos_100;
        document.getElementById('rewardGraph').src = `data:image/png;base64,${data.url_grafico}`;

        const qTableBody = document.getElementById('qTable').getElementsByTagName('tbody')[0];
        qTableBody.innerHTML = ''; 
        const numRowsToShow = Math.min(data.tabla_q.length, 16); 

        for (let i = 0; i < numRowsToShow; i++) {
            const row = qTableBody.insertRow();
            const stateCell = row.insertCell();
            stateCell.textContent = i; 

            data.tabla_q[i].forEach(value => {
                const cell = row.insertCell();
                cell.textContent = value.toFixed(4); 
                if (value > 0.5) {
                    cell.classList.add('positive');
                } else if (value < -0.5) {
                    cell.classList.add('negative');
                }
            });
        }
    })
    .catch(error => {
        document.getElementById('loadingSpinner').style.display = 'none';
        console.error('Error durante el entrenamiento:', error);
        alert('Hubo un error al entrenar el agente. Por favor, revisa la consola para más detalles.');
    });
});