class Evaluator:
    def __init__(self, model, validation_data):
        self.model = model
        self.validation_data = validation_data

    def evaluate(self):
        # Evaluate the model on the validation data
        loss, accuracy = self.model.evaluate(self.validation_data)
        return {
            'loss': loss,
            'accuracy': accuracy
        }

    def print_evaluation(self):
        evaluation_results = self.evaluate()
        print(f"Loss: {evaluation_results['loss']}, Accuracy: {evaluation_results['accuracy']}")