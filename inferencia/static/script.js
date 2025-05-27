let filaUseCases = ['Vazio', 'Vazio'];

function adicionarCasoDeUso(casoDeUso) {
    if (filaUseCases[0] === 'Vazio') {
        filaUseCases[0] = casoDeUso;
    } else if (filaUseCases[1] === 'Vazio') {
        filaUseCases[1] = casoDeUso;
    } else {
        // Move o segundo caso para a primeira posição e adiciona o novo na segunda
        filaUseCases[0] = filaUseCases[1];
        filaUseCases[1] = casoDeUso;
    }
    
    atualizarInterface();
}

function atualizarInterface() {
    document.getElementById('slot1').textContent = filaUseCases[0];
    document.getElementById('slot2').textContent = filaUseCases[1];
    
    // Habilita o botão de previsão apenas quando houver dois casos de uso
    const predictButton = document.getElementById('predict-button');
    predictButton.disabled = filaUseCases.includes('Vazio');
}

async function fazerPrevisao() {
    const predictButton = document.getElementById('predict-button');
    predictButton.disabled = true;
    
    try {
        const response = await fetch('http://localhost:8000/prever', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                casoDeUso_1: filaUseCases[0],
                casoDeUso_2: filaUseCases[1],
                periodo_mes: 'dia_folha',
                top_n: 5
            })
        });
        
        const data = await response.json();
        
        // Exibe o resultado
        const resultDiv = document.getElementById('prediction-result');
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = `
            <h3>Próximos casos de uso mais prováveis:</h3>
            <ul>
                ${Object.entries(data.previsoes)
                    .map(([caso, prob]) => `<li>${caso}: ${(prob * 100).toFixed(2)}%</li>`)
                    .join('')}
            </ul>
        `;
    } catch (error) {
        console.error('Erro ao fazer previsão:', error);
        const resultDiv = document.getElementById('prediction-result');
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = '<p style="color: red;">Erro ao fazer previsão. Verifique se a API está rodando.</p>';
    }
    
    predictButton.disabled = false;
}
