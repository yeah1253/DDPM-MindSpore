convnet_small_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [10, 20],
    'pe_dim': 128
}

convnet_medium_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [10, 10, 20, 20, 40, 40, 80, 80],
    'pe_dim': 256,
    'insert_t_to_all_layers': True
}
convnet_big_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [20, 20, 40, 40, 80, 80, 160, 160],
    'pe_dim': 256,
    'insert_t_to_all_layers': True
}

unet_1_cfg = {'type': 'UNet', 'channels': [10, 20, 40, 80], 'pe_dim': 128}
unet_res_cfg = {
    'type': 'UNet',
    'channels': [10, 20, 40, 80],
    'pe_dim': 128,
    'residual': True
}

convnet1d_big_cfg = {
    'type': 'ConvNet1D',
    'intermediate_channels': [20, 20, 40, 40, 80, 80, 160, 160, 40, 40, 10, 10, 1],
    'pe_dim': 512,
    'insert_t_to_all_layers': True
}

convnet1d_medium_cfg = {
    'type': 'ConvNet1D',
    'intermediate_channels': [20, 20, 40, 40, 80, 80, 40, 40, 10, 10, 1],
    'pe_dim': 512,
    'insert_t_to_all_layers': True
}

convnet1d_small_cfg = {
    'type': 'ConvNet1D',
    'intermediate_channels': [10, 20, 20, 10, 10, 1],
    'pe_dim': 512,
    'insert_t_to_all_layers': True
}

unet_res1d_cfg = {
    'type': 'UNet1D',
    'channels': [10, 20, 40, 80],
    'pe_dim': 128,
    'residual': True
}
unet_res1d_medium_cfg = {
    'type': 'UNet1D',
    'channels': [10, 20, 40, 80],
    'pe_dim': 512,
    'residual': True
}
unet_res1d_big_cfg = {
    'type': 'UNet1D',
    'channels': [10, 20, 40, 80, 160, 320],
    'pe_dim': 128,
    'residual': True
}

bi_lstm_small_cfg = {
    'type': 'BiLSTM',
    'lstm_hidden_size': 32,  # lstm隐藏层大小
    'pe_dim': 128,
    'num_layers': 4,
}

bi_lstm_medium_cfg = {
    'type': 'BiLSTM',
    'lstm_hidden_size': 32,
    'pe_dim': 128,
    'num_layers': 8,
}

bi_lstm_big_cfg = {
    'type': 'BiLSTM',
    'lstm_hidden_size': 32,
    'pe_dim': 128,
    'num_layers': 16,
}
convnet1d_big_classify_cfg = {
    'type': 'ConvNet1DClassify',
    'intermediate_channels': [20, 40, 80, 40, 20],
    'out_dim': 4,
}

convnet1d_medium_classify_cfg = {
    'type': 'ConvNet1DClassify',
    'intermediate_channels': [20, 40, 80, 20],
    'out_dim': 4,
}

convnet1d_small_classify_cfg = {
    'type': 'ConvNet1DClassify',
    'intermediate_channels': [10, 20, 10],
    'out_dim': 4,
}

convnet1d_mini_classify_cfg = {
    'type': 'ConvNet1DClassify',
    'intermediate_channels': [10, 10],
    'out_dim': 4,
}

configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg, unet_res_cfg,  # 0-4, 图片处理
    convnet1d_big_cfg, convnet1d_medium_cfg, convnet1d_small_cfg,  # 5-7， 信号处理
    unet_res1d_cfg, unet_res1d_medium_cfg, unet_res1d_big_cfg,  # 8-10， 信号处理
    bi_lstm_big_cfg, bi_lstm_medium_cfg, bi_lstm_small_cfg,  # 11-13， 信号处理
    convnet1d_big_classify_cfg, convnet1d_medium_classify_cfg, convnet1d_small_classify_cfg,
    convnet1d_mini_classify_cfg  # 14-17， 信号预测
]
