#!/usr/bin/env python3
"""Deploy RETGEN API to production."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import time
import argparse
import subprocess
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RETGENDeployment:
    """Handle RETGEN API deployment."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize deployment.
        
        Args:
            model_path: Path to pre-trained model
        """
        self.model_path = model_path
        self.project_root = Path(__file__).parent.parent
    
    def check_requirements(self):
        """Check deployment requirements."""
        logger.info("Checking deployment requirements...")
        
        # Check Docker
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
            logger.info("✓ Docker is installed")
        except subprocess.CalledProcessError:
            logger.error("✗ Docker is not installed")
            return False
        
        # Check Docker Compose
        try:
            subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
            logger.info("✓ Docker Compose is installed")
        except subprocess.CalledProcessError:
            logger.error("✗ Docker Compose is not installed")
            return False
        
        return True
    
    def build_image(self):
        """Build Docker image."""
        logger.info("Building Docker image...")
        
        cmd = ["docker-compose", "build", "--no-cache"]
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if result.returncode != 0:
            logger.error("Failed to build Docker image")
            return False
        
        logger.info("✓ Docker image built successfully")
        return True
    
    def deploy_local(self):
        """Deploy locally using Docker Compose."""
        logger.info("Deploying RETGEN API locally...")
        
        # Set environment variables
        env = os.environ.copy()
        if self.model_path:
            env["RETGEN_MODEL_PATH"] = f"/app/models/{self.model_path.name}"
        
        # Start services
        cmd = ["docker-compose", "up", "-d"]
        result = subprocess.run(cmd, cwd=self.project_root, env=env)
        
        if result.returncode != 0:
            logger.error("Failed to start services")
            return False
        
        logger.info("✓ Services started successfully")
        
        # Wait for API to be ready
        logger.info("Waiting for API to be ready...")
        time.sleep(10)
        
        # Check health
        try:
            import requests
            response = requests.get("http://localhost:8000/")
            if response.status_code == 200:
                logger.info("✓ API is healthy")
                return True
        except Exception as e:
            logger.error(f"API health check failed: {e}")
        
        return False
    
    def deploy_production(self, host: str, user: str = "ubuntu"):
        """Deploy to production server.
        
        Args:
            host: Production server hostname
            user: SSH user
        """
        logger.info(f"Deploying to production server {host}...")
        
        # Create deployment package
        logger.info("Creating deployment package...")
        package_path = self.project_root / "retgen-deploy.tar.gz"
        
        cmd = [
            "tar", "-czf", str(package_path),
            "--exclude", "__pycache__",
            "--exclude", "*.pyc",
            "--exclude", ".git",
            "--exclude", "models/*",
            "--exclude", "tests/*",
            "."
        ]
        subprocess.run(cmd, cwd=self.project_root, check=True)
        
        # Copy to server
        logger.info("Copying to server...")
        cmd = ["scp", str(package_path), f"{user}@{host}:~/retgen-deploy.tar.gz"]
        subprocess.run(cmd, check=True)
        
        # Deploy on server
        logger.info("Deploying on server...")
        deploy_script = """
        cd ~
        mkdir -p retgen
        cd retgen
        tar -xzf ~/retgen-deploy.tar.gz
        docker-compose down
        docker-compose build
        docker-compose up -d
        rm ~/retgen-deploy.tar.gz
        """
        
        cmd = ["ssh", f"{user}@{host}", deploy_script]
        subprocess.run(cmd, check=True)
        
        # Clean up
        package_path.unlink()
        
        logger.info(f"✓ Deployed to {host}")
        logger.info(f"API available at http://{host}")
    
    def show_logs(self):
        """Show container logs."""
        cmd = ["docker-compose", "logs", "-f", "--tail", "100"]
        subprocess.run(cmd, cwd=self.project_root)
    
    def stop_services(self):
        """Stop all services."""
        logger.info("Stopping services...")
        cmd = ["docker-compose", "down"]
        subprocess.run(cmd, cwd=self.project_root)
        logger.info("✓ Services stopped")


def main():
    """Main deployment script."""
    parser = argparse.ArgumentParser(description="Deploy RETGEN API")
    parser.add_argument(
        "--mode",
        choices=["local", "production"],
        default="local",
        help="Deployment mode"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to pre-trained model"
    )
    parser.add_argument(
        "--host",
        help="Production server hostname (for production mode)"
    )
    parser.add_argument(
        "--user",
        default="ubuntu",
        help="SSH user for production deployment"
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Show container logs"
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop services"
    )
    
    args = parser.parse_args()
    
    # Create deployment instance
    deployment = RETGENDeployment(args.model_path)
    
    # Handle commands
    if args.stop:
        deployment.stop_services()
        return
    
    if args.logs:
        deployment.show_logs()
        return
    
    # Check requirements
    if not deployment.check_requirements():
        logger.error("Deployment requirements not met")
        sys.exit(1)
    
    # Build image
    if not deployment.build_image():
        logger.error("Failed to build image")
        sys.exit(1)
    
    # Deploy
    if args.mode == "local":
        if deployment.deploy_local():
            logger.info("\n🎉 RETGEN API deployed successfully!")
            logger.info("API endpoint: http://localhost:8000")
            logger.info("Health check: http://localhost:8000/")
            logger.info("API docs: http://localhost:8000/docs")
            logger.info("\nRun 'python scripts/deploy_api.py --logs' to see logs")
        else:
            logger.error("Deployment failed")
            sys.exit(1)
    
    elif args.mode == "production":
        if not args.host:
            logger.error("--host required for production deployment")
            sys.exit(1)
        
        deployment.deploy_production(args.host, args.user)


if __name__ == "__main__":
    main()