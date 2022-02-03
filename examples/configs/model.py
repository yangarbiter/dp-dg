model_defaults = {
    'head_bert-base-uncased': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'dpall_bert-base-uncased': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'dp_bert-base-uncased': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'bert-base-uncased': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'distilbert-base-uncased': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'code-gpt-py': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'densenet121': {
        'model_kwargs': {
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'wideresnet50': {
        'model_kwargs': {
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'vgg11': {
        'model_kwargs':{
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'vgg16': {
        'model_kwargs':{
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'dp_resnet18': {
        'model_kwargs':{
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'dp_resnet50': {
        'model_kwargs':{
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'resnet18': {
        'model_kwargs':{
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'resnet34': {
        'model_kwargs':{
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'resnet50': {
        'model_kwargs': {
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'gin-virtual': {},
    'resnet18_ms': {
        'target_resolution': (224, 224),
    },
    'logistic_regression': {},
    'unet-seq': {
        'optimizer': 'Adam'
    },
    'fasterrcnn': {
        'model_kwargs': {
            'pretrained_model': True,
            'pretrained_backbone': True,
            'min_size' :1024,
            'max_size' :1024
        }
    }
}
