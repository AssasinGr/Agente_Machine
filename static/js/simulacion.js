const startSimulationBtn = document.getElementById('startSimulationBtn');
const simulationOutput = document.getElementById('simulationOutput');
const currentMapGrid = document.getElementById('currentMapGrid');
const envInfo = JSON.parse(document.getElementById('envInfoData').textContent);

let currentRow = -1;
let currentCol = -1;
let maxRows = 0;
let maxCols = 0;

if (envInfo && envInfo.map) {
    maxRows = envInfo.map.length;
    maxCols = envInfo.map[0].length;
}

function updateMap(oldState, newState) {
    if (oldState !== null && oldState !== undefined) {
        const oldRow = Math.floor(oldState / maxCols);
        const oldCol = oldState % maxCols;
        const oldCell = document.getElementById(`cell-${oldRow}-${oldCol}`);
        if (oldCell) {
            oldCell.classList.remove('agent');
            oldCell.textContent = envInfo.map[oldRow][oldCol];
        }
    }
    if (newState !== null && newState !== undefined) {
        const newRow = Math.floor(newState / maxCols);
        const newCol = Math.floor(newState % maxCols); 
        const newCell = document.getElementById(`cell-${newRow}-${newCol}`);
        if (newCell) {
            newCell.classList.add('agent');
            newCell.textContent = 'A';
            currentRow = newRow;
            currentCol = newCol;
        }
    }
}

function resetSimulationDisplay() {
    simulationOutput.innerHTML = '<p>Haz clic en "Iniciar Simulación" para ver el agente en acción.</p>';
    const currentAgentCell = currentMapGrid.querySelector('.grid-cell.agent');
    if (currentAgentCell) {
        const idParts = currentAgentCell.id.split('-');
        const oldRow = parseInt(idParts[1]);
        const oldCol = parseInt(idParts[2]);
        updateMap(oldRow * maxCols + oldCol, null);
    }
}

document.addEventListener('DOMContentLoaded', resetSimulationDisplay);

startSimulationBtn.addEventListener('click', function() {
    if (this.disabled && !document.querySelector('.warning-message')) { 
        alert('El agente no ha sido entrenado. Por favor, entrena un agente primero.');
        return;
    }
    this.disabled = true;
    simulationOutput.innerHTML = '';
    resetSimulationDisplay();

    fetch('/simular', {
        method: 'GET',
        headers: {
            'X-Requested-With': 'XMLHttpRequest'
        }
    })
    .then(response => {
        if (!response.ok) {
            return response.text().then(text => {
                try {
                    const jsonError = JSON.parse(text);
                    throw new Error(jsonError.error || `Error del servidor: ${response.status} ${response.statusText}`);
                } catch (e) {
                    throw new Error(`Error del servidor: ${response.status} ${response.statusText}. Respuesta: ${text.substring(0, 200)}...`);
                }
            });
        }
        return response.json(); 
    })
    .then(data => {
        if (data.error) {
            simulationOutput.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
            alert(data.error);
            startSimulationBtn.disabled = false;
            return;
        }
        let oldState = null;

        function animateStep(index) {
            if (index < data.path.length) {
                const step = data.path[index];

                const stepDiv = document.createElement('div');
                stepDiv.classList.add('simulation-step');
                stepDiv.innerHTML = `
                    <span class="step-number">${index + 1}.</span>
                    <div class="step-details">
                        Estado: <strong>${step.state}</strong>, Acción: <strong>${step.action}</strong>, Recompensa: <strong>${step.reward}</strong>
                        ${ envInfo && envInfo.actions_map && envInfo.actions_map[step.action_idx] ? `(${envInfo.actions_map[step.action_idx]})` : '' }
                    </div>
                `;
                simulationOutput.appendChild(stepDiv);
                simulationOutput.scrollTop = simulationOutput.scrollHeight;

                updateMap(oldState, step.state);
                oldState = step.state;

                setTimeout(() => animateStep(index + 1), 500);
            } else {
                const finalDiv = document.createElement('div');
                finalDiv.classList.add('simulation-step');
                finalDiv.innerHTML = `
                    <span class="step-number">FIN.</span>
                    <div class="step-details">
                        <p>Simulación finalizada. Recompensa Total: <strong>${data.final_reward}</strong></p>
                    </div>
                `;
                simulationOutput.appendChild(finalDiv);
                simulationOutput.scrollTop = simulationOutput.scrollHeight;

                if (data.path.length > 0) {
                    updateMap(oldState, data.path[data.path.length - 1].state);
                }


                startSimulationBtn.disabled = false;
            }
        }
        animateStep(0);
    })
    .catch(error => {
        console.error('Error durante la simulación:', error);
        simulationOutput.innerHTML = `<p style="color: red;">No se pudo iniciar la simulación. Error de red o del servidor: ${error.message}</p>`;
        startSimulationBtn.disabled = false;
    });
});