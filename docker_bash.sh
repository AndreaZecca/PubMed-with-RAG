# Insufficient number of arguments
if [ $# -lt 1 ]; then
    echo "Usage: ./run_docker.sh [run|exec|build|stop|remove]"
    exit 1
fi

case $1 in
    run)
        # Run the docker container
        docker run -v ./:/src/ --rm --gpus device=$CUDA_VISIBLE_DEVICES -d -it -p 19530:19530 --name pubmed-container pubmed
        ;;
    exec)
        # Execute the models inside the docker container
        docker exec -it pubmed-container bash      
        ;;
    build)
        # Build the docker
        docker build ./ -t pubmed
        ;;
    stop)
        # Stop the docker container
        docker stop pubmed-container
        ;;
    remove)
        # Remove the docker container
        docker stop pubmed-container &&
        docker remove pubmed-container
        ;;
    *)
        # Invalid argument
        echo "Invalid argument"
        ;;
esac
