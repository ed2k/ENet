python ENet/scripts/test_seqs.py --model ENet/prototxts/enet_deploy_final.prototxt \
    --weights ENet/enet_weights_zoo/cityscapes_weights.caffemodel  \
    --colours ENet/scripts/cityscapes19.png --input_image $1 
#python ENet/scripts/test_segmentation.py --model ENet/prototxts/enet_deploy_final.prototxt --weights ENet/enet_weights_zoo/cityscapes_weights.caffemodel  --colours ENet/scripts/cityscapes19.png --input_image ENet/example_image/munich_000000_000019_leftImg8bit.png --out_dir ENet/example_image/
