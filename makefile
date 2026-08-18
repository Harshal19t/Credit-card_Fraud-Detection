test-root:
	curl http://$$(minikube service fraud-detection-service --url)

test-health:
	curl http://$$(minikube service fraud-detection-service --url)/health

test-predict:
	curl -X POST http://$$(minikube service fraud-detection-service --url)/predict -H "Content-Type: application/json" -d '{\"features\": [1, 2, 3, 4]}'