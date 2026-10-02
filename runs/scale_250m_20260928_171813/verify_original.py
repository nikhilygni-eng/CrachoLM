import importlib.util,json,pathlib,torch
from src.inference import load_model_for_inference
torch.set_num_threads(2)
root=pathlib.Path("/home/escanor/Downloads/CrachoLM")
backup=root/".quality_backups/scale_250m_20260928_171813/src/model.py"
spec=importlib.util.spec_from_file_location("before_scaling_model",backup)
old_module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_module)
model,tok,info=load_model_for_inference(device="cpu")
old=old_module.CrachoLM(model.config).eval()
old.load_state_dict(model.state_dict(),strict=True)
ids=torch.tensor([tok.encode("User: Hi! Assistant:",add_special_tokens=False)])
with torch.inference_mode():
 before,_=old(ids)
 after,_=model(ids)
torch.testing.assert_close(before,after,atol=0,rtol=0)
result=dict(parameters=info["parameters"],checkpoint=info["checkpoint"],old_logits_identical=True,use_sdpa=model.config.use_sdpa,gradient_checkpointing=model.config.gradient_checkpointing)
print(json.dumps(result,indent=2))
(root/"runs/scale_250m_20260928_171813/original_model_compatibility.json").write_text(json.dumps(result,indent=2)+"\n")
