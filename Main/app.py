from flask import Flask, request, jsonify, render_template,flash, redirect, url_for, session, send_file
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import PyPDF2
import docx2txt
import logging
import re
import nexmo
from fpdf import FPDF


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Setup logging
logging.basicConfig(level=logging.INFO)

# Nexmo configuration
client = nexmo.Client(key='c8bb823f', secret='FOmLiWYHbfk4adFa')

creators = {
    'Ayush Shekar': '+9113217988',
    'K Shreyank': '+919597653466',
    'NagaSimha N': '+919901678694'
}

# Load and preprocess the data
df = pd.read_csv('Resume.csv')

df.fillna('', inplace=True)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['Resume_str'] = df['Resume_str'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=1000)
X_text = vectorizer.fit_transform(df['Resume_str']).toarray()

X_skills = df['Category'].apply(lambda x: len(x.split(','))).values.reshape(-1, 1)

# Dummy columns for 'experience' and 'education'
X_experience = np.zeros((df.shape[0], 1))  # Dummy column for 'experience'
X_education = np.zeros((df.shape[0], 1))  # Dummy column for 'education'

X = np.hstack((X_text, X_skills, X_experience, X_education))
y = df['Category']  # Assuming 'Category' is what we want to predict

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

def calculate_ats_score(text):
    keywords = ["python", "machine learning", "data analysis", "flask", "nlp", "wordpress", "full stack developer", "css", "data structures","sql"]
    score = 0
    word_count = len(text.split())

    for keyword in keywords:
        if keyword in text:
            score += 10

    if 500 <= word_count <= 1000:
        score += 20
    elif word_count > 1000:
        score += 10

    score = min(score, 100)
    return score

job_descriptions = pd.DataFrame({
    'Job Title': ['Data Scientist', 'Software Engineer', 'Product Manager', 'Sales Manager', 'Digital Marketing Specialist', 'Network Administrator', 'Human Resources Manager', 'Financial Analyst', 'UX/UI Designer', 'Business Analyst', 'Machine Learning Engineer', 'AI Engineer'],
    'Description': [
        '''A Data Scientist collects, analyzes, and interprets large datasets to identify patterns, trends, and insights. They build and validate predictive models, perform statistical analysis, and work with machine learning algorithms to improve data-driven decision-making. Responsibilities include data cleaning, data visualization, and communicating findings to stakeholders. Proficiency in programming languages like Python or R, and experience with data visualization tools like Tableau or Power BI are essential.''',
        '''A Software Engineer designs, develops, tests, and maintains software applications. They write clean, efficient, and maintainable code in various programming languages such as Java, C++, or Python. They collaborate with cross-functional teams to define requirements, perform code reviews, and troubleshoot/debug issues. Knowledge of software development methodologies, version control systems like Git, and familiarity with frameworks and libraries relevant to their tech stack is crucial.''',
        '''A Product Manager is responsible for the strategy, roadmap, and feature definition of a product. They gather and prioritize product and customer requirements, define the product vision, and work closely with engineering, sales, marketing, and support teams to ensure revenue and customer satisfaction goals are met. Strong analytical skills, an understanding of market trends, and the ability to communicate effectively across teams are key attributes.''',
        '''A Sales Manager leads and motivates a team of sales representatives to meet and exceed sales targets. They develop and implement strategic sales plans, analyze sales data, and forecast future sales. Responsibilities include training and coaching the sales team, building and maintaining customer relationships, and ensuring customer satisfaction. Strong leadership, excellent communication skills, and the ability to analyze market trends and competitor strategies are essential.''',
        '''A Digital Marketing Specialist plans, executes, and manages digital marketing campaigns across various online platforms. They optimize content for SEO, manage PPC campaigns, utilize social media marketing, and analyze performance metrics to drive growth. Knowledge of tools like Google Analytics, AdWords, and social media management platforms is crucial. Creativity, analytical skills, and staying updated with digital marketing trends are key attributes.''',
        '''A Network Administrator is responsible for maintaining the company’s IT network, servers, and security systems. They install, configure, and troubleshoot network hardware and software, ensure network security, and optimize network performance. Proficiency in network management tools, knowledge of networking protocols, and experience with firewalls, routers, and switches are important. Strong problem-solving skills and the ability to handle network emergencies are essential.''',
        '''A Human Resources Manager oversees the recruitment, training, and development of employees. They manage employee relations, ensure compliance with labor laws, and develop HR policies and procedures. Responsibilities include performance management, employee engagement, and handling disciplinary actions. Strong interpersonal skills, knowledge of HR software, and the ability to handle sensitive situations are crucial.''',
        '''A Financial Analyst analyzes financial data to assist in decision-making processes. They create financial models, forecast future performance, and evaluate investment opportunities. Responsibilities include preparing financial reports, analyzing market trends, and providing insights to management. Proficiency in Excel, knowledge of financial software, and strong analytical skills are essential.''',
        '''A UX/UI Designer creates user-friendly and visually appealing interfaces for websites and applications. They conduct user research, develop wireframes and prototypes, and collaborate with developers to implement designs. Knowledge of design tools like Sketch, Adobe XD, and Figma, and an understanding of user-centered design principles are crucial. Creativity, attention to detail, and the ability to understand user needs are key attributes.''',
        '''A Business Analyst works with stakeholders to identify business needs and develop solutions. They gather and document requirements, analyze processes, and ensure that IT and business teams are aligned. Responsibilities include creating detailed business analysis, outlining problems, opportunities, and solutions, and supporting project implementation. Strong analytical skills, knowledge of business process modeling, and effective communication skills are essential.''',
        '''A Machine Learning Engineer designs and implements machine learning models and algorithms to solve complex problems. They work on feature extraction, model training, and evaluation, and integrate models into production systems. Proficiency in programming languages such as Python or Java, experience with machine learning frameworks like TensorFlow or PyTorch, and knowledge of data preprocessing and model tuning are crucial.''',
        '''An AI Engineer develops and deploys artificial intelligence models and systems. They work on designing AI algorithms, training models, and applying machine learning techniques to real-world problems. Experience with AI tools and frameworks, understanding of neural networks and deep learning, and proficiency in programming languages such as Python or Java are essential. AI Engineers also collaborate with cross-functional teams to integrate AI solutions into business processes.'''
    ]
})

known_skills = set([
    "python", "machine learning", "data analysis", "flask", "nlp", "wordpress",
    "full stack developer", "css", "javascript", "sql", "html", "java", "c++",
    "r", "tableau", "power bi", "excel", "apache", "spark", "aws", "azure", "docker",
    "data structures",
])

def extract_skills(text):
    text = preprocess_text(text)
    words = set(text.split())
    skills = known_skills.intersection(words)
    return list(skills)

def suggest_job(resume_text, job_descriptions):
    vectorizer = TfidfVectorizer().fit_transform([resume_text] + job_descriptions['Description'].tolist())
    cosine_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    best_job_index = cosine_sim.argmax()
    return job_descriptions['Job Title'][best_job_index]

def predict_salary(job_title):
    salary_dict = {
        'Data Scientist': '100,000 - 150,000 USD',
        'Software Engineer': '80,000 - 120,000 USD',
        'Product Manager': '90,000 - 130,000 USD',
        'Sales Manager': '70,000 - 110,000 USD',
        'Digital Marketing Specialist': '60,000 - 100,000 USD',
        'Network Administrator': '70,000 - 90,000 USD',
        'Human Resources Manager': '70,000 - 110,000 USD',
        'Financial Analyst': '60,000 - 90,000 USD',
        'UX/UI Designer': '70,000 - 110,000 USD',
        'Business Analyst': '60,000 - 90,000 USD',
        'Machine Learning Engineer': '100,000 - 150,000 USD',
        'AI Engineer': '100,000 - 150,000 USD'
    }
    return salary_dict.get(job_title, 'N/A')

def extract_contact_info(resume_text):
    email = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text)
    phone = re.findall(r'\b(?:\+?(\d{1,3}))?[-.\s]?(\d{3})[-.\s]?(\d{3})[-.\s]?(\d{4})\b', resume_text)
    
    # Format phone numbers correctly
    formatted_phone_numbers = [''.join(number) for number in phone]
    
    return {'email': email, 'phone': formatted_phone_numbers}


def recommend_courses(missing_skills):
    # This function returns a dictionary of course websites
    course_websites = {
        'Coursera': 'https://www.coursera.org',
        'edX': 'https://www.edx.org',
        'Udemy': 'https://www.udemy.com',
        'LinkedIn Learning': 'https://www.linkedin.com/learning',
        'Pluralsight': 'https://www.pluralsight.com'
    }
    return course_websites



@app.route('/')
def home():
    return render_template('index.html')  # This will now render the previous 'start.html'

@app.route('/start')
def start():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('resume_file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        resume_text = ''
        if file.filename.endswith('.pdf'):
            try:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    resume_text += page.extract_text()
            except Exception as e:
                app.logger.error(f"Error reading PDF file: {e}")
                return jsonify({'error': 'Error reading PDF file'}), 500
        elif file.filename.endswith(('.doc', '.docx')):
            try:
                resume_text = docx2txt.process(file)
            except Exception as e:
                app.logger.error(f"Error reading DOC/DOCX file: {e}")
                return jsonify({'error': 'Error reading DOC/DOCX file'}), 500
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        resume_text = preprocess_text(resume_text)
        resume_vector = vectorizer.transform([resume_text]).toarray()
        skills_vector = np.array([len(resume_text.split(','))]).reshape(-1, 1)

        experience_vector = np.zeros((1, 1))  # Dummy value for 'experience'
        education_vector = np.zeros((1, 1))  # Dummy value for 'education'

        input_vector = np.hstack((resume_vector, skills_vector, experience_vector, education_vector))

        prediction = model.predict(input_vector)[0]
        ats_score = calculate_ats_score(resume_text)
        extracted_skills = extract_skills(resume_text)
        suggested_job = suggest_job(resume_text, job_descriptions)
        job_description = job_descriptions.loc[job_descriptions['Job Title'] == suggested_job, 'Description'].values[0]
        eligibility = ats_score >= 30
        salary_prediction = predict_salary(suggested_job)
        contact_info = extract_contact_info(resume_text)

        return redirect(url_for('results', 
                                prediction=prediction, 
                                ats_score=ats_score, 
                                skills=','.join(extracted_skills),
                                suggested_job=suggested_job,
                                job_description=job_description,
                                eligibility=eligibility,
                                salary_prediction=salary_prediction,
                                email=','.join(contact_info['email']),
                                phone=','.join(contact_info['phone'])))

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        try:
            data = request.get_json()
            name = data.get('name')
            message = data.get('message')
            full_message = f"From: {name}\nMessage: {message}"

            # Send SMS to each creator
            for creator, phone in creators.items():
                logging.info(f"Sending message to {creator} (Phone: {phone})")
                response = client.send_message({
                    'from': 'Nexmo',
                    'to': phone,
                    'text': full_message,
                })
                logging.info(f"Message response: {response}")

            return jsonify({'success': True})
        except Exception as e:
            logging.error(f"Error sending message: {e}")
            return jsonify({'success': False, 'error': str(e)})

    return render_template('contact.html')

@app.route('/results')
def results():
    prediction = request.args.get('prediction')
    ats_score = request.args.get('ats_score')
    skills = request.args.get('skills')
    suggested_job = request.args.get('suggested_job')
    job_description = request.args.get('job_description')
    eligibility = request.args.get('eligibility')
    salary_prediction = request.args.get('salary_prediction')
    email = request.args.get('email')
    phone = request.args.get('phone')

    return render_template('results.html', 
                            prediction=prediction, 
                            ats_score=ats_score, 
                            skills=skills,
                            suggested_job=suggested_job, 
                            job_description=job_description, 
                            eligibility=eligibility, 
                            salary_prediction=salary_prediction, 
                            email=email, 
                            phone=phone)

@app.route('/download_pdf')
def download_pdf():
    results = session.get('results', {})
    if not results:
        return jsonify({'error': 'No results to download'}), 400

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Resume Analysis Report", ln=True, align="C")
        pdf.ln(10)
        
        pdf.cell(200, 10, txt=f"Predicted Job Title: {results['prediction']}", ln=True)
        pdf.cell(200, 10, txt=f"ATS Score: {results['ats_score']}", ln=True)
        pdf.cell(200, 10, txt=f"Extracted Skills: {', '.join(results['skills'])}", ln=True)
        pdf.cell(200, 10, txt=f"Predicted Salary: {results['predicted_salary']}", ln=True)
        pdf.ln(10)
        
        pdf.multi_cell(200, 10, txt=f"Job Description: {results['job_description']}", align="L")
        
        pdf_file = '/mnt/data/report.pdf'
        pdf.output(pdf_file)

        return send_file(pdf_file, as_attachment=True)

    except Exception as e:
        logging.error(f"Error generating PDF: {str(e)}")
        return jsonify({'error': 'Failed to generate PDF'}), 500

@app.route('/download_csv')
def download_csv():
    results = session.get('results', {})
    if not results:
        return jsonify({'error': 'No results to download'}), 400

    try:
        csv_file = '/mnt/data/report.csv'
        df = pd.DataFrame([results])
        df.to_csv(csv_file, index=False)

        return send_file(csv_file, as_attachment=True)

    except Exception as e:
        logging.error(f"Error generating CSV: {str(e)}")
        return jsonify({'error': 'Failed to generate CSV'}), 500

# @app.route('/submit', methods=['POST'])
# def submit():
#     # Simulate processing form submission and redirect to courses page
#     return ('courses_page')

if __name__ == '__main__':
    app.run(debug=True)
