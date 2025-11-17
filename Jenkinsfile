pipeline {
    agent any

    environment {
        # Virtual environment inside workspace (no permission issues)
        VENV = "${WORKSPACE}/venv"
    }

    stages {

        stage('Checkout') {
            steps {
                echo 'Checking out the repository...'
                checkout scm
            }
        }

        stage('Setup Python Environment') {
            steps {
                echo 'Creating virtual environment and upgrading pip...'
                sh '''
                    # Remove existing venv if exists
                    rm -rf "$VENV"

                    # Create virtual environment
                    python3 -m venv "$VENV"

                    # Activate and upgrade pip, setuptools, wheel
                    . "$VENV/bin/activate"
                    pip install --upgrade pip setuptools wheel

                    # Install prebuilt SciPy wheel (avoid compilation errors)
                    pip install --force-reinstall scipy==1.16.3

                    # Install all other dependencies from requirements.txt
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Run Preprocessing') {
            steps {
                echo 'Running preprocessing script...'
                sh '''
                    . "$VENV/bin/activate"
                    python preprocess.py
                '''
            }
        }

        stage('Train & Evaluate Model') {
            steps {
                echo 'Training and evaluating model...'
                sh '''
                    . "$VENV/bin/activate"
                    python trainandevaluate.py
                '''
            }
        }

        stage('Deploy Model') {
            steps {
                echo 'Deploying model...'
                sh '''
                    . "$VENV/bin/activate"
                    python deploy.py
                '''
            }
        }

        stage('Optional: Evaluate Drift') {
            steps {
                echo 'Optional: evaluating drift...'
                sh '''
                    . "$VENV/bin/activate"
                    python evaluate_drift.py || echo "Drift script not found, skipping."
                '''
            }
        }

    }

    post {
        always {
            echo 'Pipeline finished. Cleaning up workspace if needed.'
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check the logs above.'
        }
    }
}
