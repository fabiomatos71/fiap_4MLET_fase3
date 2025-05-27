let filaUseCases = ['Vazio', 'Vazio'];

function adicionarCasoDeUso(casoDeUso) {
    if (filaUseCases[0] === 'Vazio') {
        filaUseCases[0] = casoDeUso;
    } else if (filaUseCases[1] === 'Vazio') {
        filaUseCases[1] = casoDeUso;
        // Fazer previsão automaticamente quando o segundo caso for adicionado
        fazerPrevisao();
    } else {
        // Move o segundo caso para a primeira posição e adiciona o novo na segunda
        filaUseCases[0] = filaUseCases[1];
        filaUseCases[1] = casoDeUso;
        // Fazer previsão automaticamente quando um novo caso for adicionado
        fazerPrevisao();
    }
    
    atualizarInterface();
}

function atualizarInterface() {
    document.getElementById('slot1').textContent = filaUseCases[0];
    document.getElementById('slot2').textContent = filaUseCases[1];
}

function mostrarLoading(casoDeUso) {
    const overlay = document.getElementById('loading-overlay');
    const loadingText = overlay.querySelector('.loading-text');
    loadingText.textContent = `Caso de Uso ${casoDeUso} executado. Fazendo previsão...`;
    overlay.classList.add('visible');
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function esconderLoading() {
    await sleep(1500); // Espera 1.5 segundos
    const overlay = document.getElementById('loading-overlay');
    overlay.classList.remove('visible');
}

async function fazerPrevisao() {
    // Não fazer previsão se não houver dois casos de uso
    if (filaUseCases.includes('Vazio')) {
        return;
    }

    // Mostrar loading com o último caso de uso executado
    mostrarLoading(filaUseCases[1]);
    
    try {
        const response = await fetch('http://localhost:8000/prever', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                casoDeUso_1: filaUseCases[0],
                casoDeUso_2: filaUseCases[1],
                periodo_mes: document.querySelector('input[name="periodo"]:checked').value,
                top_n: 5
            })
        });
        
        const data = await response.json();
        
        // Esconde o loading e exibe o resultado
        const resultDiv = document.getElementById('prediction-result');
        esconderLoading();
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = `
            <h3>Próximos casos de uso mais prováveis:</h3>
            <div class="use-cases-grid">
                ${Object.entries(data.previsoes)
                    .map(([caso, prob]) => `
                        <button class="use-case-button" onclick="adicionarCasoDeUso('${caso}')">
                            ${caso}<br>
                            <small>(${(prob * 100).toFixed(1)}%)</small>
                        </button>
                    `)
                    .join('')}
            </div>
        `;
    } catch (error) {
        console.error('Erro ao fazer previsão:', error);
        const resultDiv = document.getElementById('prediction-result');
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = '<p style="color: red;">Erro ao fazer previsão. Verifique se a API está rodando.</p>';
    }
    

}
