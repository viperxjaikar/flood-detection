// Global variables
let preFloodFile = null;
let postFloodFile = null;

// API base URL
const API_BASE = window.location.origin;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    setupDragAndDrop();
    checkAPIHealth();
});

/**
 * Setup drag and drop functionality for image uploads
 */
function setupDragAndDrop() {
    const dropAreas = ['preFloodDropArea', 'postFloodDropArea'];
    
    dropAreas.forEach(areaId => {
        const dropArea = document.getElementById(areaId);
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });
        
        dropArea.addEventListener('drop', handleDrop, false);
    });
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    e.currentTarget.classList.add('drag-over');
}

function unhighlight(e) {
    e.currentTarget.classList.remove('drag-over');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        const file = files[0];
        const dropAreaId = e.currentTarget.id;
        const type = dropAreaId.includes('pre') ? 'pre' : 'post';
        handleImageFile(file, type);
    }
}

/**
 * Handle image upload from file input or drag & drop
 */
function handleImageUpload(input, type) {
    const file = input.files[0];
    if (file) {
        handleImageFile(file, type);
    }
}

function handleImageFile(file, type) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }
    
    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB.');
        return;
    }
    
    // Store the file
    if (type === 'pre') {
        preFloodFile = file;
        showImagePreview(file, 'preFloodImg', 'preFloodPreview', 'preFloodPlaceholder');
    } else {
        postFloodFile = file;
        showImagePreview(file, 'postFloodImg', 'postFloodPreview', 'postFloodPlaceholder');
    }
    
    // Enable analyze button if both images are uploaded
    updateAnalyzeButton();
}

function showImagePreview(file, imgId, previewId, placeholderId) {
    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById(imgId).src = e.target.result;
        document.getElementById(previewId).classList.remove('hidden');
        document.getElementById(placeholderId).classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

function updateAnalyzeButton() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = !(preFloodFile && postFloodFile);
}

/**
 * Check API health status
 */
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        if (!data.model_loaded) {
            showError('AI model is not loaded. Please contact the administrator.');
        }
    } catch (error) {
        console.warn('Could not check API health:', error);
    }
}

/**
 * Analyze uploaded images using the AI model
 */
async function analyzeImages() {
    if (!preFloodFile || !postFloodFile) {
        alert('Please upload both pre-flood and post-flood images.');
        return;
    }
    
    const loadingIndicator = document.getElementById('loadingIndicator');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsSection = document.getElementById('resultsSection');
    
    try {
        // Show loading state
        loadingIndicator.classList.add('active');
        analyzeBtn.disabled = true;
        resultsSection.classList.add('hidden');
        
        // Prepare form data
        const formData = new FormData();
        formData.append('pre_flood', preFloodFile);
        formData.append('post_flood', postFloodFile);
        
        // Make API request
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data.results);
        } else {
            throw new Error('Analysis failed');
        }
        
    } catch (error) {
        console.error('Error analyzing images:', error);
        showError('Failed to analyze images. Please try again.');
    } finally {
        // Hide loading state
        loadingIndicator.classList.remove('active');
        analyzeBtn.disabled = false;
    }
}

/**
 * Display analysis results
 */
function displayResults(results) {
    // Update summary statistics
    document.getElementById('floodPercentage').textContent = `${results.flood_percentage}%`;
    document.getElementById('confidenceScore').textContent = `${(results.confidence_score * 100).toFixed(1)}%`;
    
    // Update risk level with appropriate styling
    const riskLevelElement = document.getElementById('riskLevel');
    riskLevelElement.textContent = results.risk_level;
    
    // Style risk level based on severity
    riskLevelElement.className = 'text-2xl font-bold';
    switch (results.risk_level) {
        case 'Low':
            riskLevelElement.classList.add('text-green-600');
            break;
        case 'Moderate':
            riskLevelElement.classList.add('text-yellow-600');
            break;
        case 'High':
            riskLevelElement.classList.add('text-orange-600');
            break;
        case 'Severe':
            riskLevelElement.classList.add('text-red-600');
            break;
    }
    
    // Display result images
    document.getElementById('resultPreFlood').src = results.images.pre_flood;
    document.getElementById('resultPostFlood').src = results.images.post_flood;
    document.getElementById('resultVisualization').src = results.images.visualization;
    
    // Show results section
    document.getElementById('resultsSection').classList.remove('hidden');
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
    });
}

/**
 * City-based flood prediction (mock implementation)
 */
async function predictCityFlood() {
    const cityInput = document.getElementById('cityInput');
    const cityName = cityInput.value.trim();
    
    if (!cityName) {
        alert('Please enter a city name.');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/predict_city`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ city_name: cityName })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayCityResult(data);
        } else {
            // Handle city not found error with helpful message
            if (data.error && data.error.includes('not found in database')) {
                const availableCities = data.error.match(/Available cities: (.+)/);
                if (availableCities) {
                    const cityList = availableCities[1].split(', ').slice(0, 10).join(', ');
                    showError(`City "${cityName}" not found. Try: ${cityList}...`);
                    return;
                }
            }
            throw new Error(data.error || 'City prediction failed');
        }
        
    } catch (error) {
        console.error('Error predicting city flood:', error);
        showError('Failed to get city flood prediction. Please try again.');
    }
}

/**
 * Display city prediction results
 */
function displayCityResult(data) {
    document.getElementById('cityName').textContent = `${data.city}, ${data.state}`;
    
    const cityRiskLevel = document.getElementById('cityRiskLevel');
    cityRiskLevel.textContent = `${data.risk_level} Risk`;
    
    // Style based on risk level
    cityRiskLevel.className = 'text-2xl font-bold mb-2';
    switch (data.risk_level) {
        case 'Low':
            cityRiskLevel.classList.add('text-green-600');
            break;
        case 'Moderate':
            cityRiskLevel.classList.add('text-yellow-600');
            break;
        case 'High':
            cityRiskLevel.classList.add('text-orange-600');
            break;
        case 'Severe':
            cityRiskLevel.classList.add('text-red-600');
            break;
    }
    
    // Create detailed information display
    const weatherInfo = data.weather_data;
    const floodMonitoring = data.flood_monitoring;
    
    document.getElementById('cityDetails').innerHTML = `
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-left">
            <div class="bg-blue-50 p-3 rounded-lg">
                <h4 class="font-semibold text-blue-800 mb-2">🌧️ Weather Analysis</h4>
                <p><strong>Season:</strong> ${weatherInfo.season}</p>
                <p><strong>Rainfall Risk:</strong> ${(weatherInfo.rainfall_probability * 100).toFixed(0)}%</p>
                <p><strong>Risk Score:</strong> ${(weatherInfo.flood_risk_score * 100).toFixed(1)}%</p>
            </div>
            <div class="bg-green-50 p-3 rounded-lg">
                <h4 class="font-semibold text-green-800 mb-2">📡 Live Monitoring</h4>
                <p><strong>Source:</strong> ${floodMonitoring.source}</p>
                <p><strong>Data Available:</strong> ${floodMonitoring.data_available ? 'Yes' : 'No'}</p>
                ${floodMonitoring.flood_areas_detected !== undefined ? 
                    `<p><strong>Active Floods:</strong> ${floodMonitoring.flood_areas_detected}</p>` : ''}
            </div>
        </div>
        <div class="mt-4 text-center">
            <p><strong>Coordinates:</strong> ${data.coordinates.lat.toFixed(4)}, ${data.coordinates.lon.toFixed(4)}</p>
            <p><strong>Last Updated:</strong> ${new Date(data.last_updated).toLocaleString()}</p>
            <div class="text-xs text-gray-500 mt-2">
                <strong>Data Sources:</strong> ${data.data_sources.join(', ')}
            </div>
        </div>
    `;
    
    document.getElementById('cityResult').classList.remove('hidden');
}

/**
 * Run demo with sample images
 */
async function runDemo() {
    try {
        const response = await fetch(`${API_BASE}/demo`);
        const data = await response.json();
        
        if (data.success) {
            displayDemoResults(data.demo_results);
        } else {
            throw new Error('Demo failed');
        }
        
    } catch (error) {
        console.error('Error running demo:', error);
        showError('Failed to run demo. Please try again.');
    }
}

/**
 * Display demo results
 */
function displayDemoResults(results) {
    const demoOutput = document.getElementById('demoOutput');
    demoOutput.innerHTML = '';
    
    results.forEach(result => {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'bg-gray-50 rounded p-3 flex justify-between items-center';
        
        const riskClass = result.risk_level === 'Low' ? 'text-green-600' : 
                         result.risk_level === 'Moderate' ? 'text-yellow-600' : 
                         result.risk_level === 'High' ? 'text-orange-600' : 'text-red-600';
        
        resultDiv.innerHTML = `
            <div>
                <span class="font-medium">${result.filename}</span>
                <span class="text-sm text-gray-600 ml-2">${result.flood_percentage}% flooded</span>
            </div>
            <span class="font-bold ${riskClass}">${result.risk_level}</span>
        `;
        
        demoOutput.appendChild(resultDiv);
    });
    
    document.getElementById('demoResults').classList.remove('hidden');
}

/**
 * Show error message
 */
function showError(message) {
    alert(`Error: ${message}`);
}

/**
 * Utility function to format file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
} 