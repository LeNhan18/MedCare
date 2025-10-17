"""
Model 2: Named Entity Recognition (NER) for Symptoms and Medical Entities
Mục tiêu: Trích xuất triệu chứng và thực thể y tế từ văn bản
"""

import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
import numpy as np
import yaml
import os
from tensorflow_addons.layers import CRF
from tensorflow_addons.losses import crf_log_likelihood
from tensorflow_addons.metrics import crf_accuracy

class SymptomExtractor:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.model_config = config['model']['symptom_extractor']
        self.entities = self.model_config['entities']
        self.max_length = self.model_config['max_length']
        self.model_type = self.model_config['model_type']
        self.label_scheme = self.model_config['label_scheme']
        
        # Tạo label mapping cho BIO scheme
        self.label_to_id = {'O': 0}  # Outside
        self.id_to_label = {0: 'O'}
        
        label_id = 1
        for entity in self.entities:
            if self.label_scheme == "BIO":
                self.label_to_id[f'B-{entity}'] = label_id
                self.id_to_label[label_id] = f'B-{entity}'
                label_id += 1
                
                self.label_to_id[f'I-{entity}'] = label_id
                self.id_to_label[label_id] = f'I-{entity}'
                label_id += 1
        
        self.num_labels = len(self.label_to_id)
        self.tokenizer = None
        self.model = None
        
    def build_model(self):
        """Xây dựng model BERT-CRF cho NER"""
        # Load pre-trained BERT
        if "phobert" in self.model_type:
            model_name = "vinai/phobert-base"
        else:
            model_name = "bert-base-multilingual-cased"
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = TFAutoModel.from_pretrained(model_name)
        
        # Input layers
        input_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name="attention_mask")
        
        # BERT embeddings
        bert_output = bert_model(input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        
        # Dropout
        sequence_output = tf.keras.layers.Dropout(0.1)(sequence_output)
        
        # Dense layer for tag prediction
        logits = tf.keras.layers.Dense(
            self.num_labels,
            activation=None,
            name='logits'
        )(sequence_output)
        
        if "crf" in self.model_type:
            # CRF layer
            crf_layer = CRF(self.num_labels, name='crf')
            output = crf_layer(logits)
            
            self.model = tf.keras.Model(
                inputs=[input_ids, attention_mask],
                outputs=output
            )
            
            # Custom loss function for CRF
            def crf_loss(y_true, y_pred):
                return -crf_log_likelihood(y_pred, y_true, sequence_lengths=tf.reduce_sum(attention_mask, axis=1))[0]
            
            # Custom accuracy function for CRF
            def crf_acc(y_true, y_pred):
                return crf_accuracy(y_true, y_pred, sequence_lengths=tf.reduce_sum(attention_mask, axis=1))
            
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                loss=crf_loss,
                metrics=[crf_acc]
            )
        else:
            # Simple softmax output
            output = tf.keras.layers.Softmax()(logits)
            
            self.model = tf.keras.Model(
                inputs=[input_ids, attention_mask],
                outputs=output
            )
            
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        return self.model
    
    def preprocess_text(self, texts, labels=None):
        """Tiền xử lý văn bản và labels cho NER"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='tf',
            is_split_into_words=False
        )
        
        if labels is not None:
            # Convert labels to ids
            label_ids = []
            for label_seq in labels:
                label_id_seq = [self.label_to_id.get(label, 0) for label in label_seq]
                # Pad or truncate to max_length
                if len(label_id_seq) > self.max_length:
                    label_id_seq = label_id_seq[:self.max_length]
                else:
                    label_id_seq.extend([0] * (self.max_length - len(label_id_seq)))
                label_ids.append(label_id_seq)
            
            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': tf.constant(label_ids)
            }
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
    
    def predict(self, text):
        """Dự đoán entities từ văn bản"""
        if self.model is None:
            raise ValueError("Model chưa được load. Hãy gọi load_model() trước.")
        
        # Preprocess
        inputs = self.preprocess_text(text)
        
        # Predict
        predictions = self.model.predict([inputs['input_ids'], inputs['attention_mask']])
        
        # Decode predictions
        if "crf" in self.model_type:
            # CRF output
            predicted_ids = predictions[0]
        else:
            # Softmax output
            predicted_ids = np.argmax(predictions[0], axis=-1)
        
        # Convert back to tokens and labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        predicted_labels = [self.id_to_label[id] for id in predicted_ids]
        
        # Extract entities
        entities = self._extract_entities(tokens, predicted_labels)
        
        return {
            'tokens': tokens,
            'labels': predicted_labels,
            'entities': entities
        }
    
    def _extract_entities(self, tokens, labels):
        """Trích xuất entities từ BIO labels"""
        entities = []
        current_entity = None
        current_tokens = []
        
        for token, label in zip(tokens, labels):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
                
            if label.startswith('B-'):
                # Kết thúc entity hiện tại (nếu có)
                if current_entity:
                    entities.append({
                        'entity': current_entity,
                        'text': self.tokenizer.convert_tokens_to_string(current_tokens),
                        'tokens': current_tokens
                    })
                
                # Bắt đầu entity mới
                current_entity = label[2:]  # Bỏ 'B-'
                current_tokens = [token]
                
            elif label.startswith('I-') and current_entity == label[2:]:
                # Tiếp tục entity hiện tại
                current_tokens.append(token)
                
            else:
                # Kết thúc entity hiện tại (nếu có)
                if current_entity:
                    entities.append({
                        'entity': current_entity,
                        'text': self.tokenizer.convert_tokens_to_string(current_tokens),
                        'tokens': current_tokens
                    })
                    current_entity = None
                    current_tokens = []
        
        # Kết thúc entity cuối cùng (nếu có)
        if current_entity:
            entities.append({
                'entity': current_entity,
                'text': self.tokenizer.convert_tokens_to_string(current_tokens),
                'tokens': current_tokens
            })
        
        return entities
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, epochs=5):
        """Huấn luyện model"""
        if self.model is None:
            self.build_model()
        
        # Preprocess training data
        train_inputs = self.preprocess_text(train_texts, train_labels)
        
        if val_texts is not None and val_labels is not None:
            val_inputs = self.preprocess_text(val_texts, val_labels)
            validation_data = (
                [val_inputs['input_ids'], val_inputs['attention_mask']], 
                val_inputs['labels']
            )
        else:
            validation_data = None
        
        # Train
        history = self.model.fit(
            [train_inputs['input_ids'], train_inputs['attention_mask']],
            train_inputs['labels'],
            validation_data=validation_data,
            epochs=epochs,
            batch_size=16,
            verbose=1
        )
        
        return history
    
    def save_model(self, path=None):
        """Lưu model"""
        if path is None:
            path = self.model_config['path']
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        
        # Save tokenizer và mappings
        tokenizer_path = path.replace('.h5', '_tokenizer')
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save label mappings
        import json
        mappings_path = path.replace('.h5', '_mappings.json')
        with open(mappings_path, 'w', encoding='utf-8') as f:
            json.dump({
                'label_to_id': self.label_to_id,
                'id_to_label': self.id_to_label
            }, f, ensure_ascii=False, indent=2)
    
    def load_model(self, path=None):
        """Load model đã huấn luyện"""
        if path is None:
            path = self.model_config['path']
        
        if os.path.exists(path):
            # Load model
            custom_objects = {}
            if "crf" in self.model_type:
                custom_objects = {'CRF': CRF}
            
            self.model = tf.keras.models.load_model(path, custom_objects=custom_objects)
            
            # Load tokenizer
            tokenizer_path = path.replace('.h5', '_tokenizer')
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            
            # Load label mappings
            import json
            mappings_path = path.replace('.h5', '_mappings.json')
            if os.path.exists(mappings_path):
                with open(mappings_path, 'r', encoding='utf-8') as f:
                    mappings = json.load(f)
                    self.label_to_id = mappings['label_to_id']
                    self.id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}
        else:
            print(f"Model file không tồn tại: {path}")
            self.build_model()


# Example usage và test
if __name__ == "__main__":
    # Test examples
    test_examples = [
        "Tôi bị đau đầu và sốt nhẹ",
        "Con tôi 5 tuổi bị ho và chảy nước mũi",
        "Paracetamol có thể điều trị đau bụng không?"
    ]
    
    # Initialize extractor
    extractor = SymptomExtractor()
    extractor.build_model()
    
    print("Model Symptom Extractor đã được khởi tạo!")
    print(f"Entities: {extractor.entities}")
    print(f"Label mappings: {extractor.label_to_id}")
    print(f"Model summary:")
    extractor.model.summary()