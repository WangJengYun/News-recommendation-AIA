import tensorflow as tf 
from bert import modeling

def create_model(bert_config, is_training, 
                input_ids, input_mask, segment_ids, labels , num_labels, 
                use_one_hot_embeddings):
    
    # building Bert model
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)
    
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable('output_weights',[num_labels,hidden_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    
    output_bias = tf.get_variable('output_bias',[num_labels],initializer=tf.zeros_initializer())

    with tf.variable_scope('loss'):
        if is_training : 
            output_layer = tf.nn.dropout(output_layer,keep_prob=0.9)
        
        logits = tf.matmul(output_layer,output_weights,transpose_b = True)
        logits = tf.nn.bias_add(logits,output_bias)

        probabilities = tf.nn.sigmoid(logits)
        
        #labels = tf.case(labels,tf.float32)
        labels = tf.to_float(labels)
        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(per_example_loss)

        tf.logging.info("num_labels:{};logits:{};labels:{}".format(num_labels, logits, labels))
        
        return (loss, per_example_loss, logits, probabilities)
        

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):

    def model_fn(features, labels, mode, params):   

        input_ids = features['input_ids']
        input_mask = features['input_mask']
        segment_ids = features['segment_ids']
        label_ids = features['label_ids']
        
        is_real_example = None
        if "is_real_example" in features:
             is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
             is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
        
        print(label_ids)
        
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        
        if init_checkpoint:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        if mode == tf.estimator.ModeKeys.TRAIN:
            
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)#,
                #scaffold=scaffold_fn)
        return output_spec
    
    return model_fn

