#!/bin/bash
# Start Kafka and Zookeeper using Docker Compose

echo "Starting Kafka and Zookeeper with Docker Compose..."

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Use docker compose (newer) or docker-compose (older)
if command -v docker &> /dev/null && docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "Error: docker-compose not found"
    exit 1
fi

# Start services
$COMPOSE_CMD up -d

# Wait for Kafka to be ready
echo "Waiting for Kafka to be ready..."
sleep 10

# Create topic if it doesn't exist
echo "Creating Kafka topic 'stream-data'..."
docker exec -it kafka bash -c "kafka-topics --create \
    --topic stream-data \
    --bootstrap-server localhost:9092 \
    --partitions 1 \
    --replication-factor 1" \
    2>/dev/null || echo "Topic 'stream-data' may already exist"

echo ""
echo "Kafka and Zookeeper are running!"
echo "Kafka is available at localhost:9092"
echo ""
echo "To stop services, run: ./stop_kafka.sh"
echo "To view logs, run: docker-compose logs -f"

