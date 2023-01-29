from layers import *

"""Preparing photos"""

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 512  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

# style_img = image_loader("./resources/styles/Picasso.jpg")
# content_img = image_loader("./resources/content/dasha_face.jpg")



def tensorToImage(tensor):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    return image


def showTensor(tensor):
    image = tensorToImage(tensor)
    plt.imshow(image)


"""## Define neccessary layers"""


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


"""## Creating a new model and sending image through it"""

class NSTnet:
    def __init__(self, style_img, content_img,
                 mean=cnn_normalization_mean, std=cnn_normalization_std,
                 content_layers=content_layers, style_layers=style_layers):
        '''With basic neural net as VGG-19'''
        self.content_losses, self.style_losses = [], []
        self.style = style_img.clone()
        self.content = content_img.clone()
        self.input = content_img.clone()
        self.model = nn.Sequential(Normalization(mean, std))
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

            self.model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = self.model(content_img).detach()
                content_loss = ContentLoss(target)
                self.model.add_module("content_loss_{}".format(i), content_loss)
                self.content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = self.model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                self.model.add_module("style_loss_{}".format(i), style_loss)
                self.style_losses.append(style_loss)

            # now we trim off the layers after the last content and style losses
        for i in range(len(self.model) - 1, -1, -1):
            if isinstance(self.model[i], ContentLoss) or isinstance(self.model[i], StyleLoss):
                break

        self.model = self.model[:(i + 1)]

    def transfer_style(self, bot, user_id, num_steps=300, style_weight=100000, content_weight=3):
        self.input.requires_grad_(True)
        self.model.requires_grad_(False)

        optimizer = optim.LBFGS([self.input])
        for_gif = []
        run = [0]
        while run[0] <= num_steps:
            def closure():
                # correct the values of updated input image
                with torch.no_grad():
                    self.input.clamp_(0, 1)

                optimizer.zero_grad()
                self.model(self.input)
                style_score = 0
                content_score = 0

                for sl in self.style_losses:
                    style_score += sl.loss
                for cl in self.content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1

                if run[0] % 10 == 0:
                    for_gif.append(tensorToImage(self.input.clone()))

                if run[0] % 30 == 0:
                    bot.send_message(user_id, f'Your photo ready on {run[0]}%')

                print(f'{run[0]} / {num_steps}')

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        with torch.no_grad():
            self.input.clamp_(0, 1)

        for_gif.append(tensorToImage(self.input.clone()))
        return for_gif


def run_transfer(bot, user_id, style_path, content_path):
    style_img = image_loader(style_path)
    content_img = image_loader(content_path)
    solver = NSTnet(style_img, content_img)
    gif = solver.transfer_style(bot, user_id)
    return gif


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# def transfer_all_styles(content_img, mean, std, path=path['result'],
#                         content_layers=content_layers, style_layers=style_layers):
#     styles = os.listdir(path['style'])
#     counter = 0
#     for st in styles:
#         if st.startswith('.'):
#             continue
#         style_img = image_loader(path['style'] + st)
#         net = NSTnet(style_img, content_img, mean, std, content_layers, style_layers)
#         gif = net.transfer_style(num_steps=500)
#         save_pic_and_gif(gif, path + str(counter))
#         counter += 1