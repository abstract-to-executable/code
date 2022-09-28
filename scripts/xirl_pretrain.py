import time
import numpy as np
import pickle
import torch
import os
from absl import logging

from skilltranslation.models.xirl_models import LinearEncoderNet, Resnet18LinearEncoderNet
from skilltranslation.data.xirl_pretrain_dataset import SkillTranslationDataset
from skilltranslation.training.tcc import TCCTrainer
from cfgs.xirl.configs.xmagical.pretraining.tcc_boxpusher import get_config
config = get_config()
device = torch.device("cuda")
np.random.seed(0)
torch.manual_seed(0)
ds = SkillTranslationDataset("./datasets/boxpusher_v2/dataset_xirl_randomgoal.pkl",
                             frames_per_sequence=config.frame_sampler.num_frames_per_sequence, image=False)
dl = torch.utils.data.DataLoader(
    dataset=ds, batch_size=config.data.batch_size, collate_fn=ds.collate_fn)
model = LinearEncoderNet(state_dims=6, input_type="state", embedding_size=config.model.embedding_size, normalize_embeddings=config.model.normalize_embeddings, learnable_temp=config.model.learnable_temp, num_ctx_frames=config.frame_sampler.num_context_frames)
# model = Resnet18LinearEncoderNet(embedding_size=config.model.embedding_size, normalize_embeddings=config.model.normalize_embeddings,
#                                  learnable_temp=config.model.learnable_temp, num_ctx_frames=config.frame_sampler.num_context_frames)
optim = torch.optim.Adam(
    model.parameters(),
    lr=config.optim.lr,
    weight_decay=config.optim.weight_decay,
)
trainer = TCCTrainer(model, optim, device, config)
global_step = 0
epoch = 0
complete = False

for i in range(1000):
    if complete: break
    for batch in dl:
        stime = time.time()
        train_loss = trainer.train_one_iter(batch)
        if not global_step % config.logging_frequency:
            # for k, v in train_loss.items():
            # 	logger.log_scalar(v, global_step, k, "pretrain")
            # logger.flush()
            pass
        if not global_step % config.eval.eval_frequency:
            # Evaluate the model on the pretraining validation dataset.
            valid_loss = trainer.eval_num_iters(
                dl,
                config.eval.val_iters,
            )
            print(valid_loss)
            state_dict = model.state_dict()
            torch.save(state_dict, "model.pt")
            # for k, v in valid_loss.items():
            #     logger.log_scalar(v, global_step, k, "pretrain")

        # Evaluate the model on the downstream datasets.
        # for split, downstream_loader in downstream_loaders.items():
        #     eval_to_metric = eval_manager.evaluate(
        #         model,
        #         downstream_loader,
        #         device,
        #         config.eval.val_iters,
        #     )
        #     for eval_name, eval_out in eval_to_metric.items():
        #         eval_out.log(
        #             logger,
        #             global_step,
        #             eval_name,
        #             f"downstream/{split}",
        #         )

        # Save model checkpoint.
        # if not global_step % config.checkpointing_frequency:
        # 	checkpoint_manager.save(global_step)

        # Exit if complete.
        global_step += 1
        if global_step > config.optim.train_max_iters:
            complete = True
            break
        time_per_iter = time.time() - stime
        if global_step % 50 == 0:
            print(
                "Iter[{}/{}] (Epoch {}), {:.6f}s/iter, Loss: {:.3f}".format(
                    global_step,
                    config.optim.train_max_iters,
                    epoch,
                    time_per_iter,
                    train_loss["train/total_loss"].item(),
                ))
state_dict = model.state_dict()
torch.save(state_dict, "model.pt")
