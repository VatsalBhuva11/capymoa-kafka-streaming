"""
Kafka Producer for streaming datasets (Electricity and Bike) to Kafka topics.
Supports controlled streaming rate and optional concept drift injection.
"""
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from capymoa.datasets import Electricity, Bike
import json
import time
import argparse
import sys


def create_topic_if_not_exists(bootstrap_servers, topic_name, num_partitions=1, replication_factor=1):
    """Create Kafka topic if it doesn't exist."""
    admin_client = KafkaAdminClient(
        bootstrap_servers=bootstrap_servers,
        client_id='producer_admin'
    )
    
    try:
        topic = NewTopic(
            name=topic_name,
            num_partitions=num_partitions,
            replication_factor=replication_factor
        )
        admin_client.create_topics([topic])
        print(f"Created topic: {topic_name}")
    except Exception as e:
        if "TopicExistsException" in str(e) or "TopicAlreadyExistsException" in str(e):
            print(f"Topic {topic_name} already exists")
        else:
            print(f"Error creating topic: {e}")


def stream_electricity_dataset(producer, topic_name, stream_rate=0.1, inject_drift=False, drift_point=20000):
    """
    Stream Electricity dataset (classification) to Kafka.
    
    Args:
        producer: KafkaProducer instance
        topic_name: Kafka topic name
        stream_rate: Delay between messages in seconds
        inject_drift: Whether to inject artificial drift
        drift_point: Instance number where drift should occur
    """
    print("Loading Electricity dataset...")
    stream = Electricity()
    schema = stream.get_schema()
    
    print(f"Dataset: Electricity (Classification)")
    print(f"Schema: {schema}")
    print(f"Streaming to topic: {topic_name}")
    print(f"Stream rate: {stream_rate} seconds per instance")
    if inject_drift:
        print(f"Drift will be injected at instance {drift_point}")
    print("-" * 60)
    
    instance_count = 0
    drift_triggered = False
    
    try:
        for instance in stream:
            instance_count += 1
            
            # Inject drift by swapping class labels after drift_point
            if inject_drift and instance_count >= drift_point and not drift_triggered:
                print(f"\n[DRIFT INJECTED] Swapping class labels at instance {instance_count}")
                drift_triggered = True
            
            # Extract features and label
            features = instance.x
            # Try to get label - CapyMOA instances may use y_index or y_label
            try:
                label = instance.y_index if hasattr(instance, 'y_index') else instance.y
            except:
                label = instance.y
            
            # Debug: Check label values for first few instances
            if instance_count <= 10:
                label_str = instance.y_label if hasattr(instance, 'y_label') else 'N/A'
                print(f"PRODUCER DEBUG Instance {instance_count}: label={label} (type: {type(label).__name__}), y_label={label_str}, y_index={instance.y_index if hasattr(instance, 'y_index') else 'N/A'}")
            
            # Swap labels if drift is injected
            if inject_drift and drift_triggered:
                # For binary classification, swap UP/DOWN
                label = 1 - label if label in [0, 1] else label
            
            # Create message
            message = {
                'instance_id': instance_count,
                'features': features.tolist() if hasattr(features, 'tolist') else list(features),
                'label': int(label),
                'dataset': 'electricity',
                'task': 'classification',
                'drift_injected': drift_triggered
            }
            
            # Send to Kafka
            producer.send(topic_name, message)
            
            if instance_count % 1000 == 0:
                print(f"Sent {instance_count} instances...")
            
            time.sleep(stream_rate)
            
    except KeyboardInterrupt:
        print(f"\nStopped streaming. Total instances sent: {instance_count}")
    except Exception as e:
        print(f"Error streaming data: {e}")
        raise
    
    print(f"\nFinished streaming {instance_count} instances from Electricity dataset")


def stream_bike_dataset(producer, topic_name, stream_rate=0.1, inject_drift=False, drift_point=8000):
    """
    Stream Bike dataset (regression) to Kafka.
    
    Args:
        producer: KafkaProducer instance
        topic_name: Kafka topic name
        stream_rate: Delay between messages in seconds
        inject_drift: Whether to inject artificial drift
        drift_point: Instance number where drift should occur
    """
    print("Loading Bike dataset...")
    stream = Bike()
    schema = stream.get_schema()
    
    print(f"Dataset: Bike (Regression)")
    print(f"Schema: {schema}")
    print(f"Streaming to topic: {topic_name}")
    print(f"Stream rate: {stream_rate} seconds per instance")
    if inject_drift:
        print(f"Drift will be injected at instance {drift_point}")
    print("-" * 60)
    
    instance_count = 0
    drift_triggered = False
    
    try:
        for instance in stream:
            instance_count += 1
            
            # Inject drift by scaling target values after drift_point
            if inject_drift and instance_count >= drift_point and not drift_triggered:
                print(f"\n[DRIFT INJECTED] Scaling target values at instance {instance_count}")
                drift_triggered = True
            
            # Extract features and target
            features = instance.x
            target = instance.y
            
            # Scale target if drift is injected (multiply by 2 to simulate concept drift)
            if inject_drift and drift_triggered:
                target = target * 2.0
            
            # Create message
            message = {
                'instance_id': instance_count,
                'features': features.tolist() if hasattr(features, 'tolist') else list(features),
                'target': float(target),
                'dataset': 'bike',
                'task': 'regression',
                'drift_injected': drift_triggered
            }
            
            # Send to Kafka
            producer.send(topic_name, message)
            
            if instance_count % 1000 == 0:
                print(f"Sent {instance_count} instances...")
            
            time.sleep(stream_rate)
            
    except KeyboardInterrupt:
        print(f"\nStopped streaming. Total instances sent: {instance_count}")
    except Exception as e:
        print(f"Error streaming data: {e}")
        raise
    
    print(f"\nFinished streaming {instance_count} instances from Bike dataset")


def main():
    parser = argparse.ArgumentParser(description='Stream datasets to Kafka')
    parser.add_argument('--dataset', type=str, choices=['electricity', 'bike', 'both'], 
                       default='both', help='Dataset to stream')
    parser.add_argument('--topic-prefix', type=str, default='ml-stream',
                       help='Prefix for Kafka topic names')
    parser.add_argument('--stream-rate', type=float, default=0.1,
                       help='Delay between messages in seconds')
    parser.add_argument('--inject-drift', action='store_true',
                       help='Inject artificial concept drift')
    parser.add_argument('--bootstrap-servers', type=str, default='localhost:9092',
                       help='Kafka bootstrap servers')
    
    args = parser.parse_args()
    
    # Initialize Kafka producer
    try:
        producer = KafkaProducer(
            bootstrap_servers=args.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3
        )
        print(f"Connected to Kafka at {args.bootstrap_servers}")
    except Exception as e:
        print(f"Failed to connect to Kafka: {e}")
        print("Make sure Kafka is running (docker-compose up)")
        sys.exit(1)
    
    # Stream datasets
    if args.dataset in ['electricity', 'both']:
        topic_name = f"{args.topic_prefix}-electricity"
        create_topic_if_not_exists(args.bootstrap_servers, topic_name)
        stream_electricity_dataset(
            producer, 
            topic_name, 
            stream_rate=args.stream_rate,
            inject_drift=args.inject_drift
        )
    
    if args.dataset in ['bike', 'both']:
        topic_name = f"{args.topic_prefix}-bike"
        create_topic_if_not_exists(args.bootstrap_servers, topic_name)
        stream_bike_dataset(
            producer, 
            topic_name, 
            stream_rate=args.stream_rate,
            inject_drift=args.inject_drift
        )
    
    producer.flush()
    producer.close()
    print("Producer closed.")


if __name__ == '__main__':
    main()

