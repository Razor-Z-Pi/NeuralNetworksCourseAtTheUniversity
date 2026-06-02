// Глобальные переменные для графиков
let datasetPlot = null;
let heatmapPlot = null;
let lossChart = null;
let accuracyChart = null;
let currentGrid = null;
let currentData = null;

// Инициализация графиков при загрузке
document.addEventListener('DOMContentLoaded', function() {
    initializePlots();
    setupEventListeners();
});

/**
 * Инициализация пустых графиков Plotly
 */
function initializePlots() {
    const datasetLayout = {
        title: 'Выберите параметры и сгенерируйте датасет',
        xaxis: { title: 'X₁', range: [-1.5, 1.5] },
        yaxis: { title: 'X₂', range: [-1.5, 1.5] },
        showlegend: true,
        plot_bgcolor: '#fafafa',
        paper_bgcolor: '#fafafa',
        height: 350,
        margin: { t: 40, r: 20, b: 40, l: 40 }
    };
    
    Plotly.newPlot('dataset-plot', [], datasetLayout);
    
    const heatmapLayout = {
        title: 'Предсказания модели',
        xaxis: { title: 'X₁', range: [-1.5, 1.5] },
        yaxis: { title: 'X₂', range: [-1.5, 1.5] },
        plot_bgcolor: '#fafafa',
        paper_bgcolor: '#fafafa',
        height: 350,
        margin: { t: 40, r: 20, b: 40, l: 40 }
    };
    
    Plotly.newPlot('heatmap-plot', [], heatmapLayout);
    
    // График потерь
    const lossCtx = document.getElementById('loss-chart').getContext('2d');
    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Loss',
                data: [],
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
    
    // График точности
    const accCtx = document.getElementById('accuracy-chart').getContext('2d');
    accuracyChart = new Chart(accCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Test Accuracy',
                data: [],
                borderColor: '#43e97b',
                backgroundColor: 'rgba(67, 233, 123, 0.1)',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { 
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

/**
 * Настройка обработчиков событий
 */
function setupEventListeners() {
    document.getElementById('noise').addEventListener('input', function(e) {
        document.getElementById('noise-value').textContent = e.target.value;
    });
    
    document.getElementById('n-samples').addEventListener('input', function(e) {
        document.getElementById('samples-value').textContent = e.target.value;
    });
    
    document.getElementById('hidden-size').addEventListener('input', function(e) {
        document.getElementById('hidden-value').textContent = e.target.value;
    });
    
    document.getElementById('learning-rate').addEventListener('input', function(e) {
        document.getElementById('lr-value').textContent = e.target.value;
    });
    
    document.getElementById('epochs').addEventListener('input', function(e) {
        document.getElementById('epochs-value').textContent = e.target.value;
    });
    
    // Кнопка генерации датасета
    document.getElementById('generate-btn').addEventListener('click', generateDataset);
    
    // Кнопка обучения
    document.getElementById('train-btn').addEventListener('click', trainNetwork);
    
    // Кнопка сброса
    document.getElementById('reset-btn').addEventListener('click', resetAll);
}

/**
 * Генерация датасета через API
 */
async function generateDataset() {
    const statusDiv = document.getElementById('status');
    statusDiv.innerHTML = '<span class="loading"></span> Генерация датасета...';
    
    const params = {
        dataset_type: document.getElementById('dataset-type').value,
        noise: parseFloat(document.getElementById('noise').value),
        n_samples: parseInt(document.getElementById('n-samples').value)
    };
    
    try {
        const response = await fetch('/api/generate_dataset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        
        const data = await response.json();
        currentData = data;
        
        visualizeDataset(data);
        
        statusDiv.innerHTML = 'Датасет сгенерирован!!! Нажмите "Обучить сеть"';
        
        // Очистка старых предсказаний
        clearHeatmap();
        clearCharts();
        
    } catch (error) {
        statusDiv.innerHTML = 'Ошибка при генерации датасета!!!';
        console.error('Error:', error);
    }
}

/**
 * Визуализация датасета на графике Plotly
 */
function visualizeDataset(data) {
    const X = data.X;
    const y = data.y;
    
    // Разделение точек по классам
    const class0 = { x: [], y: [] };
    const class1 = { x: [], y: [] };
    
    for (let i = 0; i < X.length; i++) {
        if (y[i] === 0) {
            class0.x.push(X[i][0]);
            class0.y.push(X[i][1]);
        } else {
            class1.x.push(X[i][0]);
            class1.y.push(X[i][1]);
        }
    }
    
    const traces = [
        {
            x: class0.x,
            y: class0.y,
            mode: 'markers',
            type: 'scatter',
            name: 'Класс 0',
            marker: {
                color: '#667eea',
                size: 8,
                line: { color: '#555', width: 1 }
            }
        },
        {
            x: class1.x,
            y: class1.y,
            mode: 'markers',
            type: 'scatter',
            name: 'Класс 1',
            marker: {
                color: '#f5576c',
                size: 8,
                line: { color: '#555', width: 1 }
            }
        }
    ];
    
    const layout = {
        title: 'Распределение классов в датасете',
        xaxis: { title: 'X₁', range: [-1.5, 1.5] },
        yaxis: { title: 'X₂', range: [-1.5, 1.5] },
        showlegend: true,
        plot_bgcolor: '#fafafa',
        paper_bgcolor: '#fafafa',
        height: 350
    };
    
    Plotly.react('dataset-plot', traces, layout);
}

async function trainNetwork() {
    if (!currentData) {
        alert('Сначала сгенерируйте датасет!!!');
        return;
    }
    
    const statusDiv = document.getElementById('status');
    statusDiv.innerHTML = '<span class="loading"></span> Обучение сети... Это может занять несколько секунд';
    
    const params = {
        hidden_size: parseInt(document.getElementById('hidden-size').value),
        learning_rate: parseFloat(document.getElementById('learning-rate').value),
        epochs: parseInt(document.getElementById('epochs').value)
    };
    
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        
        const data = await response.json();
        
        visualizeHeatmap(data.grid);
        updateLossChart(data.losses);
        updateAccuracyChart(data.accuracies);
        updateTrainingInfo(data);
        
        statusDiv.innerHTML = 'Обучение завершено!!!';
        
    } catch (error) {
        statusDiv.innerHTML = 'Ошибка при обучении сети!!!';
        console.error('Error:', error);
    }
}

/**
 * Визуализация тепловой карты предсказаний
 */
function visualizeHeatmap(grid) {
    currentGrid = grid;
    
    // Создание сетки координат
    const x = [];
    const y = [];
    const z = [];
    
    const gridSize = grid.length;
    
    for (let i = 0; i < gridSize; i++) {
        const row = [];
        for (let j = 0; j < gridSize; j++) {
            row.push(grid[i][j]);
            if (i === 0) {
                x.push(-1 + (2 * j) / (gridSize - 1));
            }
        }
        y.push(-1 + (2 * i) / (gridSize - 1));
        z.push(row);
    }
    
    const trace = {
        z: z,
        x: x,
        y: y,
        type: 'heatmap',
        colorscale: [
            [0, '#667eea'],
            [1, '#f5576c']
        ],
        showscale: true,
        colorbar: {
            title: 'Класс',
            tickvals: [0, 1],
            ticktext: ['0', '1']
        }
    };
    
    const layout = {
        title: 'Граница решений',
        xaxis: { title: 'X₁', range: [-1, 1] },
        yaxis: { title: 'X₂', range: [-1, 1] },
        plot_bgcolor: '#fafafa',
        paper_bgcolor: '#fafafa',
        height: 350
    };
    
    Plotly.react('heatmap-plot', [trace], layout);
    
    addContourToDataset(grid);
}

/**
 * Добавление контура решений на график датасета
 */
function addContourToDataset(grid) {
    if (!currentData) return;
    
    visualizeDataset(currentData);
    
    const gridSize = grid.length;
    const x = [];
    const y = [];
    
    for (let i = 0; i < gridSize; i++) {
        x.push(-1 + (2 * i) / (gridSize - 1));
        y.push(-1 + (2 * i) / (gridSize - 1));
    }
    
    const contourTrace = {
        z: grid,
        x: x,
        y: y,
        type: 'contour',
        contours: {
            start: 0.5,
            end: 0.5,
            coloring: 'lines'
        },
        line: {
            color: 'black',
            width: 3
        },
        showscale: false
    };
    
    Plotly.addTraces('dataset-plot', contourTrace);
}

/**
 * Обновление графика потерь
 */
function updateLossChart(losses) {
    lossChart.data.labels = Array.from({length: losses.length}, (_, i) => i + 1);
    lossChart.data.datasets[0].data = losses;
    lossChart.update();
}

/**
 * Обновление графика точности
 */
function updateAccuracyChart(accuracies) {
    accuracyChart.data.labels = Array.from({length: accuracies.length}, (_, i) => i + 1);
    accuracyChart.data.datasets[0].data = accuracies;
    accuracyChart.update();
}

/**
 * Очистка тепловой карты
 */
function clearHeatmap() {
    Plotly.react('heatmap-plot', [], {
        title: 'Ожидание обучения...',
        xaxis: { title: 'X₁' },
        yaxis: { title: 'X₂' },
        plot_bgcolor: '#fafafa',
        paper_bgcolor: '#fafafa'
    });
}

/**
 * Очистка графиков обучения
 */
function clearCharts() {
    lossChart.data.labels = [];
    lossChart.data.datasets[0].data = [];
    lossChart.update();
    
    accuracyChart.data.labels = [];
    accuracyChart.data.datasets[0].data = [];
    accuracyChart.update();
}

/**
 * Обновление информационной панели
 */
function updateTrainingInfo(data) {
    const finalLoss = data.losses[data.losses.length - 1].toFixed(4);
    const finalAcc = (data.accuracies[data.accuracies.length - 1] * 100).toFixed(2);
    
    const infoDiv = document.getElementById('training-info');
    infoDiv.innerHTML = `
        <p><strong>Финальные потери (Training Loss):</strong> ${finalLoss}</p>
        <p><strong>Финальная точность (Test Accuracy):</strong> ${finalAcc}%</p>
        <p><strong>Эпох обучения:</strong> ${data.losses.length}</p>
        <hr>
        <p style="color: #666; font-size: 0.9em;">
            <em>Чем ближе потери к 0 и точность к 100%, тем лучше модель разделяет классы!!!</em>
        </p>
    `;
}

/**
 * Сброс всех настроек
 */
function resetAll() {
    clearHeatmap();
    clearCharts();
    Plotly.react('dataset-plot', [], {
        title: 'Ожидание генерации датасета',
        xaxis: { title: 'X₁' },
        yaxis: { title: 'X₂' }
    });
    
    document.getElementById('status').innerHTML = 'Система сброшена. Сгенерируйте новый датасет!!!';
    document.getElementById('training-info').innerHTML = '<p>Для начала работы сгенерируйте датасет и нажмите "Обучить сеть"!!!</p>';
    
    currentData = null;
    currentGrid = null;
}