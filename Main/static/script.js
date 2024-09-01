document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('resume-form');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');
    

    console.log('DOM fully loaded and parsed');

    if (form) {
        form.onsubmit = function (event) {
            event.preventDefault();
            console.log('Form submitted');

            const formData = new FormData(form);
            console.log('Form data:', formData);

            loadingDiv.style.display = 'block';  // Show loading message
            errorDiv.style.display = 'none';     // Hide error message

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                console.log('Response data:', data);
                loadingDiv.style.display = 'none';  // Hide loading message
                
                if (data.error) {
                    errorDiv.innerText = data.error;
                    errorDiv.style.display = 'block';
                    return;
                }

                // If on the results page, populate the results
                if (window.location.pathname === '/results') {
                    populateResults(data);
                } else {
                    // Redirect to results page with query parameters
                    const queryParams = new URLSearchParams(data).toString();
                    console.log('Redirecting to /results with queryParams:', queryParams);
                    window.location.href = '/results?' + queryParams;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                loadingDiv.style.display = 'none';  // Hide loading message
                errorDiv.innerText = 'An error occurred: ' + error.message;
                errorDiv.style.display = 'block';
            });
        };
    }

    // If on the results page, populate the results from query parameters
    if (window.location.pathname === '/results') {
        populateResultsFromQueryParams();
    }
});

function populateResults(data) {
    document.getElementById('prediction').innerText = data.prediction || 'N/A';
    document.getElementById('ats_score').innerText = data.ats_score || 'N/A';
    document.getElementById('skills').innerText = data.skills ? data.skills.join(', ') : 'N/A';
    document.getElementById('suggested_job').innerText = data.suggested_job || 'N/A';
    document.getElementById('job_description').innerText = data.job_description || 'N/A';
    document.getElementById('eligibility').innerText = data.eligibility ? 'Yes' : 'No';
    document.getElementById('salary_prediction').innerText = data.salary_prediction || 'N/A';
    document.getElementById('email').innerText = data.contact_info.email ? data.contact_info.email.join(', ') : 'N/A';
    document.getElementById('phone').innerText = data.contact_info.phone ? data.contact_info.phone.join(', ') : 'N/A';
}

function populateResultsFromQueryParams() {
    const urlParams = new URLSearchParams(window.location.search);

    console.log('URL Params:', urlParams.toString());
    document.getElementById('prediction').innerText = urlParams.get('prediction') || 'N/A';
    document.getElementById('ats_score').innerText = urlParams.get('ats_score') || 'N/A';
    document.getElementById('skills').innerText = urlParams.get('skills') ? urlParams.get('skills').split(',').join(', ') : 'N/A';
    document.getElementById('suggested_job').innerText = urlParams.get('suggested_job') || 'N/A';
    document.getElementById('job_description').innerText = urlParams.get('job_description') || 'N/A';
    document.getElementById('eligibility').innerText = urlParams.get('eligibility') === 'true' ? 'Yes' : 'No';
    document.getElementById('salary_prediction').innerText = urlParams.get('salary_prediction') || 'N/A';
    document.getElementById('email').innerText = urlParams.get('email') ? urlParams.get('email').split(',').join(', ') : 'N/A';
    document.getElementById('phone').innerText = urlParams.get('phone') ? urlParams.get('phone').split(',').join(', ') : 'N/A';
}
