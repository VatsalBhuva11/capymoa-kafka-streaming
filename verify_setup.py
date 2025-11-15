"""
Verification script to check if the setup is correct.
Tests Kafka connection, CapyMOA installation, and dataset loading.
"""
import sys
import subprocess

def check_kafka():
    """Check if Kafka is running."""
    print("Checking Kafka connection...")
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers='localhost:9092',
            request_timeout_ms=5000
        )
        producer.close()
        print("✓ Kafka is accessible")
        return True
    except Exception as e:
        print(f"✗ Kafka connection failed: {e}")
        print("  Make sure Kafka is running: docker-compose up -d")
        return False

def check_capymoa():
    """Check if CapyMOA is installed and working."""
    print("\nChecking CapyMOA installation...")
    try:
        import capymoa
        print(f"✓ CapyMOA version: {capymoa.__version__ if hasattr(capymoa, '__version__') else 'installed'}")
        
        # Test dataset loading
        from capymoa.datasets import Electricity, Bike
        print("  Testing Electricity dataset...")
        elec = Electricity()
        schema = elec.get_schema()
        print(f"  ✓ Electricity schema: {schema}")
        
        print("  Testing Bike dataset...")
        bike = Bike()
        schema = bike.get_schema()
        print(f"  ✓ Bike schema: {schema}")
        
        return True
    except ImportError as e:
        print(f"✗ CapyMOA not installed: {e}")
        print("  Install with: pip install capymoa")
        return False
    except Exception as e:
        print(f"✗ CapyMOA error: {e}")
        return False

def check_dependencies():
    """Check if all required packages are installed."""
    print("\nChecking dependencies...")
    required_packages = [
        'kafka',
        'numpy',
        'pandas',
        'sklearn',
        'matplotlib',
        'seaborn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'kafka':
                __import__('kafka')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n  Install missing packages: pip install {' '.join(missing)}")
        return False
    return True

def check_docker():
    """Check if Docker is available."""
    print("\nChecking Docker...")
    try:
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"  ✓ {result.stdout.strip()}")
            return True
        else:
            print("  ✗ Docker not found")
            return False
    except FileNotFoundError:
        print("  ✗ Docker not installed or not in PATH")
        return False
    except Exception as e:
        print(f"  ✗ Error checking Docker: {e}")
        return False

def main():
    print("=" * 60)
    print("Streaming ML Pipeline Setup Verification")
    print("=" * 60)
    
    all_ok = True
    
    # Check Docker
    docker_ok = check_docker()
    all_ok = all_ok and docker_ok
    
    # Check dependencies
    deps_ok = check_dependencies()
    all_ok = all_ok and deps_ok
    
    # Check CapyMOA
    capymoa_ok = check_capymoa()
    all_ok = all_ok and capymoa_ok
    
    # Check Kafka
    kafka_ok = check_kafka()
    all_ok = all_ok and kafka_ok
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All checks passed! Setup is ready.")
        print("\nYou can now:")
        print("  1. Start Kafka: ./start_kafka.sh or docker-compose up -d")
        print("  2. Run producer: python producer.py --dataset electricity")
        print("  3. Run consumer: python consumer.py --topic ml-stream-electricity --task classification --model hoeffding_tree")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

