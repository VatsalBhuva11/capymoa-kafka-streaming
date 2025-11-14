#!/bin/bash
# Stop Kafka and Zookeeper Docker containers

echo "Stopping Kafka and Zookeeper..."

# Use docker compose (newer) or docker-compose (older)
if command -v docker &> /dev/null && docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "Error: docker-compose not found"
    exit 1
fi

$COMPOSE_CMD down

echo "Kafka and Zookeeper stopped."

