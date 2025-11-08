pipeline {
  agent any
  parameters {
    string(name: 'PROJECT_NAME', defaultValue: 'diabetes', description: 'Project name')
    booleanParam(name: 'AUTO_DEPLOY', defaultValue: false, description: 'Auto promote to Production')
  }
  environment {
    OMP_NUM_THREADS = '2'
    MKL_NUM_THREADS = '2'
    OPENBLAS_NUM_THREADS = '2'
    NUM_WORKER_THREADS = '3'
    MLFLOW_TRACKING_URI = "${env.MLFLOW_TRACKING_URI ?: 'http://localhost:5001'}"
    PUSHGATEWAY_URL = "${env.PUSHGATEWAY_URL ?: 'http://localhost:9091'}"
  }
  stages {
    stage('Checkout') { steps { checkout scm } }

    stage('Build Docker image') {
      steps { sh 'docker build -t mlops-agp:${BUILD_NUMBER} .' }
    }

    stage('Preprocess') {
      steps {
        sh "docker run --rm -v ${pwd()}:/app -w /app -e DATA_DIR=/app/data -e OMP_NUM_THREADS=${env.OMP_NUM_THREADS} -e MKL_NUM_THREADS=${env.MKL_NUM_THREADS} -e OPENBLAS_NUM_THREADS=${env.OPENBLAS_NUM_THREADS} mlops-agp:${BUILD_NUMBER} python preprocess.py"
      }
    }

    stage('Train & Evaluate') {
      steps {
        sh "docker run --rm -v ${pwd()}:/app -w /app -e MLFLOW_TRACKING_URI=${env.MLFLOW_TRACKING_URI} -e PUSHGATEWAY_URL=${env.PUSHGATEWAY_URL} -e NUM_WORKER_THREADS=${env.NUM_WORKER_THREADS} -e OMP_NUM_THREADS=${env.OMP_NUM_THREADS} -e MKL_NUM_THREADS=${env.MKL_NUM_THREADS} -e OPENBLAS_NUM_THREADS=${env.OPENBLAS_NUM_THREADS} mlops-agp:${BUILD_NUMBER} python trainandevaluate.py"
        archiveArtifacts artifacts: 'models/last_run_summary.json', onlyIfSuccessful: true
      }
    }

    stage('Validate') {
      steps {
        // put your validation script here (example placeholder)
        sh "echo 'Validation placeholder - implement scripts/validate_model.py if needed'"
      }
    }

    stage('Deploy Staging') {
      steps {
        sh "docker run --rm -v ${pwd()}:/app -w /app -e MLFLOW_TRACKING_URI=${env.MLFLOW_TRACKING_URI} mlops-agp:${BUILD_NUMBER} python deploy.py --project ${params.PROJECT_NAME} --stage Staging"
      }
    }

    stage('Promote to Production') {
      when { expression { return params.AUTO_DEPLOY } }
      steps {
        sh "docker run --rm -v ${pwd()}:/app -w /app -e MLFLOW_TRACKING_URI=${env.MLFLOW_TRACKING_URI} mlops-agp:${BUILD_NUMBER} python deploy.py --project ${params.PROJECT_NAME} --stage Production"
      }
    }
  }
  post {
    success { echo 'Pipeline finished successfully' }
    failure { echo 'Pipeline failed' }
  }
}
