import os
import time
import torch

class SolverWrapper(object):
    def __init__(self, params):
        self.max_epoch = params.max_epoch
        self.print_freq = params.print_freq
        self.validate_interval = params.validate_interval
        self.save_interval = params.save_interval
        self.experiment = params.experiment
        self.best_checkpoint_path = os.path.join(self.experiment, 'lstm_ctc_demon.pth')

    def train(self, train_loader, val_loader, model,
              criterion, optimizer, device, converter):
        print('Start training ...')
        is_best = False
        step = 0
        best_accuracy = 0.0

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        model.train()  # switch to train mode
        end = time.time()

        for epoch in range(self.max_epoch):
            """training in one epoch."""
            for datum in train_loader:
                # measure data loading time
                batch_size = datum[0].shape[0]
                data_time.update(time.time() - end, batch_size)
                loss = self._train_step(datum, model, criterion, optimizer, device, converter)
                # measure elapsed time
                batch_time.update(time.time() - end, batch_size)
                losses.update(loss.item(), batch_size)
                step += 1

                # print_freq
                if step % self.print_freq == 0:
                    print('=> epoch: {:>3}, step: {:>6}, '
                          'time: {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                          'loss = {loss.val:.4f} ({loss.avg:.4f})'.format(
                            epoch, step, batch_time=batch_time, loss=losses))

                # validate
                if step % self.validate_interval == 0:
                    print('=> training data time: {:.6f}'.format(data_time.avg))
                    print('=> Evaluating on validation dataset ...')
                    accuracy, duration = self._validate(val_loader, model, device, converter)
                    print('=> duration: {:.3f}, '
                          'current accuracy = {:.4f}, '
                          'best accuracy = {:.4f}'.format(
                            duration, accuracy, best_accuracy))
                    if (accuracy >= best_accuracy and accuracy > 0.0):
                        best_accuracy = accuracy
                        print('==> Saving the model to {}'.format(self.best_checkpoint_path))
                        self._save_checkpoint(model, self.best_checkpoint_path, is_best=True)
                    model.train()  # switch to train mode
                # reset the time
                end = time.time()
                if step % self.save_interval == 0:
                    checkpoint_path = os.path.join(self.experiment,
                                                   'lstm_ctc_{}_{}.pth'.format(epoch, step))
                    print('==> Saving model to {}'.format(checkpoint_path))
                    self._save_checkpoint(model, checkpoint_path)
            # reset the losses
            losses.reset()
            batch_time.reset()
            data_time.reset()
        print('Done!')

    @staticmethod
    def _train_step(datum, model, criterion, optimizer, device, converter):
        """Train."""
        images, labels = datum
        batch_size, seq_len = images.shape[:2]
        # step 1. Clear out gradients
        model.zero_grad()
        # step 2. Get our inputs images ready for the network.
        preds_size = torch.IntTensor(batch_size).fill_(seq_len)
        # clear out hidden state of the LSTM
        model.hidden = model.init_hidden(batch_size)

        # labels is a list of `torch.InTensor` with `batch_size` size.
        labels, lengths = converter.encode(labels)
        # to(device)
        images = images.to(device)

        # step 3. Run out forward pass.
        preds = model(images)

        # step 4. Compute the loss, gradients, and update the parameters
        # by calling optimizer.step()
        loss = criterion(preds, labels, preds_size, lengths) / batch_size
        loss.backward()
        optimizer.step()
        return loss

    @staticmethod
    def _validate(val_loader, model, device, converter):
        """Validate."""
        duration = time.time()
        model.eval()  # switch to evaluate mode
        num_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                batch_size = images.shape[0]
                model.hidden = model.init_hidden(batch_size)
                images = images.to(device)
                outputs = model(images)
                preds = converter.predict(outputs)
                for pred, label in zip(preds, labels):
                    if pred == label:
                        num_correct += 1
        accuracy = num_correct / len(val_loader.dataset)
        duration = time.time() - duration
        return accuracy, duration

    def _save_checkpoint(self, model, checkpoint_path, is_best=False):
        if is_best:
            if os.path.isfile(self.best_checkpoint_path):
                os.remove(self.best_checkpoint_path)
        torch.save(model.state_dict(), checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
