import tensorflow as tf
from   keras.layers.core import Lambda
import keras.backend  as K
from   keras.layers import Subtract,Concatenate,Dot,Reshape,Permute,Conv1D

# df = train_input;bert_V = train_bert
def input_data_for_model(df,bert_V,mclass_V,feat_dict,is_predict = False):
    input_data_1 = [df[feat.name].values for feat in feat_dict['cat']]
    input_data_2 = [df[feat.name].values for feat in feat_dict['cont']]
    if feat_dict['bert']:
        input_data3 = [bert_V.values]
    else:
        input_data3 = []
    
    if feat_dict['mclass']:
        input_data4 = [mclass_V.values]
    else:
        input_data4 = []

    input_data = input_data_1 + input_data_2 + input_data3 +input_data4
                
    if not is_predict:
        target = df['label'].values
    else :
        target = None
    return input_data , target

def model_input(feat_dict):
    cat_input_dict = OrderedDict()
    for feat in feat_dict['cat']:
        cat_input_dict[feat.name] = Input(shape=(1,), name='cat_'+feat.name, dtype=feat.dtype)

    cont_input_dict = OrderedDict()
    for feat in feat_dict["cont"]:
            cont_input_dict[feat.name] = Input(shape=(feat.dimension,), name='cont_'+feat.name, dtype=feat.dtype)
    if feat_dict["bert"]:
        Bert_feat = feat_dict["bert"]
        Bert_Input = [Input(shape=(Bert_feat.dimension,), name='bert_'+Bert_feat.name, dtype=Bert_feat.dtype)]
    else:
        Bert_Input = []

    if feat_dict["mclass"]:
        mclass_feat = feat_dict["mclass"]
        mclass_Input = [Input(shape=(mclass_feat.dimension,), name='mclass_'+ mclass_feat.name, dtype=mclass_feat.dtype)]
    else:
        mclass_Input = []

    inputs_list = list(cat_input_dict.values()) + list(cont_input_dict.values()) + Bert_Input + mclass_Input
    
    return cat_input_dict,cont_input_dict,Bert_Input,mclass_Input,inputs_list

def embedding_layer(cat_input_dict,cont_input_dict,Bert_Input,mclass_Input,embedding_size):
    Deep_input_list = []
    # setting embedding 
    deep_cat_emb_dict = dict()
    for feat in feat_dict['cat']:
        deep_cat_emb_dict[feat.name] = Embedding(feat.dimension,embedding_size,\
            #embeddings_regularizer=l2(l2_reg_embedding),
            #embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std, seed=seed),
            name = 'emb_cat_' + feat.name)

    embedding_vec_list = []
    for feat in feat_dict['cat']:
            lookup_idx = cat_input_dict[feat.name]
            feat_vec = deep_cat_emb_dict[feat.name](lookup_idx)
            embedding_vec_list.append(feat_vec)

    Deep_input_list.extend(embedding_vec_list)
    #---------
    #cont 
    if cont_input_dict:
        if len(cont_input_dict) != 1:
            all_other_cont = Concatenate()(list(cont_input_dict.values()))
        else :
            all_other_cont = list(cont_input_dict.values())[0]
        cont_Dense_list_1 = [Dense(embedding_size)(all_other_cont)]
        cont_Dense_list_2 = list(map(Reshape((1, embedding_size)), cont_Dense_list_1))

        Deep_input_list.extend(cont_Dense_list_2)
    
    if Bert_Input:
        #Bert_layer_1 = Dense(384,activation=Activation('relu'))(Bert_Input[0])
        Bert_layer_1 = Dense(384)(Bert_Input[0])
        Bert_layer_2 = Dense(embedding_size)(Bert_layer_1)
        Bert_layer_final = Reshape((1, embedding_size))(Bert_layer_2)
        Deep_input_list.extend([Bert_layer_final])
    
    if mclass_Input:
        mclass_feat = feat_dict['mclass']
        mclass_embedding = Embedding(101,embedding_size,\
            input_length=mclass_feat.dimension,
            #embeddings_regularizer=l2(l2_reg_embedding),
            #embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std, seed=seed),
            name = 'emb_mclass_' + mclass_feat.name)(mclass_Input[0])
        Deep_input_list.extend([mclass_embedding])

    return Deep_input_list

def linear_embedding_layer(cat_input_dict,cont_input_dict,Bert_Input,mclass_Input):
    Deep_input_list = []
    # setting embedding 
    deep_cat_emb_dict = dict()
    for feat in feat_dict['cat']:
        deep_cat_emb_dict[feat.name] = Embedding(feat.dimension,1,\
            #embeddings_regularizer=l2(l2_reg_embedding),
            #embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std, seed=seed),
            name = 'linear_cat_' + feat.name)

    embedding_vec_list = []
    for feat in feat_dict['cat']:
            lookup_idx = cat_input_dict[feat.name]
            feat_vec = deep_cat_emb_dict[feat.name](lookup_idx)
            embedding_vec_list.append(feat_vec)

    Deep_input_list.extend(embedding_vec_list)
    #---------
    #cont 
    if cont_input_dict:
        cont_Dense_list = list(map(Reshape((1, 1)), list(cont_input_dict.values())))
        Deep_input_list.extend(cont_Dense_list)
    
    if Bert_Input:
        Bert_layer_1 = Dense(1)(Bert_Input[0])
        Bert_layer_final = Reshape((1, 1))(Bert_layer_1)
        Deep_input_list.extend([Bert_layer_final])
    
    if mclass_Input:
        mclass_feat = feat_dict['mclass']
        mclass_embedding_1 = Embedding(101,1,\
            input_length=mclass_feat.dimension,
            #embeddings_regularizer=l2(l2_reg_embedding),
            #embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std, seed=seed),
            name = 'linear_mclass_' + mclass_feat.name)(mclass_Input[0])
        mclass_embedding = Lambda(lambda z: K.sum(z, axis=1,keepdims=True))(mclass_embedding_1)
        Deep_input_list.extend([mclass_embedding])

    linear_term  = Reshape((1,))(Add()(Deep_input_list))

    return linear_term

# inputs = Deep_input_1
def FM(inputs) :

    concated_embeds_value = inputs
    square_of_sum = Lambda(lambda z: K.square(z))(Lambda(lambda z: K.sum(z, axis=1,keepdims=True))(concated_embeds_value))
    sum_of_square = Lambda(lambda z: K.sum(z, axis=1,keepdims=True))(Lambda(lambda z: K.square(z))(concated_embeds_value))   
    cross_term = Subtract()([square_of_sum,sum_of_square])
    cross_term = Lambda(lambda z: 0.5*z)(Lambda(lambda z: K.sum(z, axis=2,keepdims=False))(cross_term))
    
    return cross_term

def CIN(inputs,layer_size):
    dim = inputs.get_shape()[-1].value
    hidden_nn_layers  = [inputs]
    final_result = []

    split_tensor0  = Lambda( lambda x: tf.split(x,dim,2))(hidden_nn_layers[0])
    field_nums0 = hidden_nn_layers[0].get_shape()[1].value

    for idx,onelayer_size in enumerate(layer_size) :
        # onelayer_size = layer_size[0]
        split_tensor = Lambda( lambda x: tf.split(x,dim,2))(hidden_nn_layers[-1])
        field_nums = hidden_nn_layers[-1].get_shape()[1].value

        dot_result_m = Lambda(lambda x,split_tensor0: tf.matmul(split_tensor0,x, transpose_b=True),\
            arguments={'split_tensor0':split_tensor0})(split_tensor)

        dot_result_o  = Lambda(lambda x: tf.reshape(x,shape=[dim,-1,field_nums0*field_nums]))(dot_result_m)

        dot_result   = Lambda(lambda x: tf.transpose(x,perm=[1,0,2]))(dot_result_o)

        curr_out  = Conv1D(onelayer_size,1, strides=1, padding='valid')(dot_result)

        curr_out    = Lambda(lambda x: tf.transpose(x,perm=[0,2,1]))(curr_out )

        final_result.append(curr_out)
        hidden_nn_layers.append(curr_out)

    result = Concatenate(axis=1)(final_result)

    result  = Lambda(lambda z: K.sum(z, axis=-1))(result)

    #result = Dense(1)(result)

    return result















#----------------------------------
split_half  =False

import tensorflow as tf
import tensorflow.keras.layers.Layer as add_weight

field_nums = [inputs.get_shape()[1].value]
filters = []
bias = []
for i, size in enumerate(layer_size):
    filters.append(add_weight(name='filter' + str(i),
                                        shape=[1, field_nums[-1]
                                               * field_nums[0], size],
                                        dtype=tf.float32))
    bias.append(add_weight(name='bias' + str(i), shape=[size], dtype=tf.float32,
                                     initializer=tf.keras.initializers.Zeros()))
    if split_half :
        if i != len(layer_size) - 1 and size % 2 > 0:
            raise ValueError(
                "layer_size must be even number except for the last layer when split_half=True")
        field_nums.append(size // 2)
    else:
        field_nums.append(size)