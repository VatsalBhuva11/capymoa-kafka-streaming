"""
Setup Verification Script
Verifies that all dependencies and services are properly configured.
"""

import sys
import subprocess

def check_docker():
    """Check if Docker is installed and running."""
    print("Checking Docker...")
    try:
        # Check if docker command exists
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"  ✓ Docker installed: {result.stdout.strip()}")
        else:
            print("  ✗ Docker not found")
            return False
        
        # Check if Docker daemon is running
        result = subprocess.run(['docker', 'ps'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("  ✓ Docker daemon is running")
            return True
        else:
            print("  ✗ Docker daemon is not running")
            print("    Start Docker Desktop or Docker service")
            return False
    except FileNotFoundError:
        print("  ✗ Docker not installed")
        print("    Install Docker: https://docs.docker.com/get-docker/")
        return False
    except Exception as e:
        print(f"  ✗ Error checking Docker: {e}")
        return False

def check_kafka_docker():
    """Check if Kafka Docker containers are running."""
    print("\nChecking Kafka Docker containers...")
    try:
        # Check if containers are running
        result = subprocess.run(['docker', 'compose', 'ps'], 
                              capture_output=True, text=True, timeout=5, cwd='.')
        if result.returncode != 0:
            # Try docker-compose (older version)
            result = subprocess.run(['docker-compose', 'ps'], 
                                  capture_output=True, text=True, timeout=5, cwd='.')
        
        if 'kafka' in result.stdout.lower() and 'Up' in result.stdout:
            print("  ✓ Kafka container is running")
            return True
        else:
            print("  ✗ Kafka container is not running")
            print("    Start with: ./start_kafka.sh")
            return False
    except FileNotFoundError:
        print("  ✗ docker-compose not found")
        return False
    except Exception as e:
        print(f"  ✗ Error checking containers: {e}")
        return False

def check_imports():
    """Check if all required packages are installed."""
    print("Checking Python packages...")
    packages = {
        'capymoa': 'capymoa',
        'kafka': 'kafka-python',
        'river': 'river',
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'plotly': 'plotly',
        'dash': 'dash',
        'dash_bootstrap_components': 'dash-bootstrap-components'
    }
    
    missing = []
    capymoa_error = None
    
    for module, package in packages.items():
        try:
            # For capymoa, use a subprocess to avoid crashing the main process
            if module == 'capymoa':
                import subprocess
                result = subprocess.run(
                    [sys.executable, '-c', f'import {module}'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode != 0:
                    error_msg = result.stderr[:100] if result.stderr else "Unknown error"
                    if "SIGSEGV" in error_msg or "fatal error" in error_msg.lower():
                        print(f"  ⚠ {package} - INSTALLED but Java crash detected")
                        print("     This is a JPype1/Java compatibility issue.")
                        capymoa_error = "Java crash (SIGSEGV)"
                    elif "jpype" in error_msg.lower() or "java" in error_msg.lower():
                        print(f"  ⚠ {package} - INSTALLED but JPype error")
                        capymoa_error = error_msg
                    else:
                        print(f"  ✗ {package} - Import error")
                        missing.append(package)
                else:
                    print(f"  ✓ {package}")
            else:
                __import__(module)
                print(f"  ✓ {package}")
        except ImportError as e:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
        except RuntimeError as e:
            # Handle capymoa JPype errors
            if module == 'capymoa':
                print(f"  ⚠ {package} - INSTALLED but JPype error: {str(e)[:60]}...")
                print("     Note: CapyMOA requires Java and JPype1. This may affect dataset loading.")
                capymoa_error = str(e)
            else:
                print(f"  ✗ {package} - Runtime error: {str(e)[:60]}...")
                missing.append(package)
        except Exception as e:
            print(f"  ✗ {package} - Error: {str(e)[:60]}...")
            missing.append(package)
    
    return len(missing) == 0, missing, capymoa_error

def check_kafka():
    """Check if Kafka is accessible."""
    print("\nChecking Kafka connection...")
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers='localhost:9092',
            request_timeout_ms=5000
        )
        # Try to get metadata
        metadata = producer.list_topics(timeout=5)
        producer.close()
        print("  ✓ Kafka is accessible")
        return True
    except Exception as e:
        print(f"  ✗ Kafka connection failed: {e}")
        print("    Make sure Kafka is running on localhost:9092")
        return False

def check_datasets():
    """Check if datasets can be loaded."""
    print("\nChecking dataset loading...")
    try:
        from capymoa.datasets import Electricity, Covtype, Sensor, Fried, Bike
        
        datasets = {
            'Electricity': Electricity,
            'Covtype': Covtype,
            'Sensor': Sensor,
            'Fried': Fried,
            'Bike': Bike
        }
        
        for name, dataset_class in datasets.items():
            try:
                dataset = dataset_class()
                # Try to get first instance
                instance = next(iter(dataset))
                print(f"  ✓ {name}")
            except Exception as e:
                print(f"  ✗ {name} - Error: {str(e)[:80]}")
                return False
        
        return True
    except ImportError as e:
        print(f"  ✗ Cannot import datasets: {e}")
        return False
    except RuntimeError as e:
        print(f"  ✗ CapyMOA runtime error (likely JPype/Java issue): {str(e)[:80]}")
        print("     Install Java: sudo apt-get install default-jdk  # Linux")
        print("     Or: brew install openjdk  # macOS")
        print("     Then reinstall: pip install --upgrade --force-reinstall capymoa jpype1")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False

def main():
    print("="*60)
    print("Streaming ML Pipeline - Setup Verification")
    print("="*60)
    
    # Check Docker
    docker_ok = check_docker()
    docker_compose_ok = check_kafka_docker() if docker_ok else False
    
    # Check imports
    imports_ok, missing, capymoa_error = check_imports()
    
    # Check Kafka connection
    kafka_ok = check_kafka() if docker_compose_ok else False
    
    # Check datasets (skip if capymoa has import issues)
    datasets_ok = False
    if not capymoa_error:
        datasets_ok = check_datasets()
    else:
        print("\nSkipping dataset check due to CapyMOA import error")
    
    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    if docker_ok and docker_compose_ok and imports_ok and kafka_ok and datasets_ok:
        print("✓ All checks passed! You're ready to run the pipeline.")
        return 0
    else:
        print("✗ Some checks failed:")
        if not docker_ok:
            print("  - Docker is not installed or not running")
            print("    Install Docker: https://docs.docker.com/get-docker/")
        elif not docker_compose_ok:
            print("  - Kafka Docker containers are not running")
            print("    Start with: ./start_kafka.sh")
        if not imports_ok:
            print(f"  - Missing packages: {', '.join(missing)}")
            print(f"    Install with: pip install {' '.join(missing)}")
        if capymoa_error:
            print("  - CapyMOA has JPype/Java issues")
            if "crash" in capymoa_error.lower() or "SIGSEGV" in capymoa_error:
                print("    Java is crashing - this is a compatibility issue.")
                print("    Try these solutions:")
                print("    1. Use a different Java version:")
                print("       sudo pacman -S jdk17-openjdk  # or jdk11-openjdk")
                print("       export JAVA_HOME=/usr/lib/jvm/java-17-openjdk")
                print("    2. Reinstall JPype1 with specific version:")
                print("       pip install --upgrade --force-reinstall 'jpype1>=1.4.0'")
                print("    3. Set JPype environment variable:")
                print("       export JPYPE_JVM_PATH=/usr/lib/jvm/java-17-openjdk/lib/server/libjvm.so")
                print("    4. Alternative: Use synthetic data generators instead of CapyMOA datasets")
            else:
                print("    Install Java: sudo pacman -S jdk-openjdk  # Arch Linux")
                print("    Or: sudo apt-get install default-jdk  # Debian/Ubuntu")
                print("    Or: brew install openjdk  # macOS")
                print("    Then: pip install --upgrade --force-reinstall capymoa jpype1")
        if not kafka_ok and docker_compose_ok:
            print("  - Kafka is not accessible")
            print("    Check containers: docker compose ps")
            print("    Restart: ./stop_kafka.sh && ./start_kafka.sh")
        if not datasets_ok and not capymoa_error:
            print("  - Dataset loading failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())

