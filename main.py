import torchvision.models as models
import torch

cnn = models.vgg19(pretrained=True)
cnn = next(cnn.children())[:14]
torch.save(cnn, './vgg19_result.pth')

"""def make_save(style_img, content_img,
             mean=cnn_normalization_mean, std=cnn_normalization_std,
             content_layers=content_layers, style_layers=style_layers):
    '''With basic neural net as VGG-19'''
    content_losses, style_losses = [], []
    style = style_img.clone()
    content = content_img.clone()
    input = content_img.clone()
    model = nn.Sequential(Normalization(mean, std))
    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model


style_img = image_loader('/resources/styles/Malevich.jpg')
content_ing = image_loader('/resources/content/Angelina.jpg')
model = make_save(, )

torch.save(model, './')
"""