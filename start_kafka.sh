#!/bin/bash
# Start Kafka and Zookeeper using Docker Compose

echo "Starting Kafka and Zookeeper..."
docker-compose up -d

echo "Waiting for Kafka to be ready..."
sleep 10

echo "Checking Kafka status..."
docker-compose ps

echo ""
echo "Kafka is running at localhost:9092"
echo "Zookeeper is running at localhost:2181"

