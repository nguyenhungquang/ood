python test_ood_detection.py run_name=cifar10_wrn detector=msp ckpt_path=checkpoints/pretrained/cifar10_wrn_pretrained_epoch_99.pt test_bs=64 
python test_ood_detection.py run_name=cifar10_wrn detector=energy ckpt_path=checkpoints/pretrained/cifar10_wrn_pretrained_epoch_99.pt test_bs=64 
python test_ood_detection.py run_name=cifar10_wrn detector=ODIN ckpt_path=checkpoints/pretrained/cifar10_wrn_pretrained_epoch_99.pt test_bs=64 
python test_ood_detection.py run_name=cifar10_wrn detector=mahalanobis ckpt_path=checkpoints/pretrained/cifar10_wrn_pretrained_epoch_99.pt test_bs=64 
python test_ood_detection.py run_name=cifar10_wrn detector=pNML ckpt_path=checkpoints/pretrained/cifar10_wrn_pretrained_epoch_99.pt test_bs=64 

# Multi-run
python test_ood_detection.py -m run_name=cifar10_wrn detector=msp,ODIN,energy,mahalanobis,pNML ckpt_path=checkpoints/pretrained/cifar10_wrn_pretrained_epoch_99.pt test_bs=64 
