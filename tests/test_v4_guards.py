import unittest
import numpy as np
import torch
from training.data_v21 import Sequences, audit
from training.robustness_v4 import perturbation
from models.multitask import CausalConv1d, TCNCausalBaseline, TransformerCausalBaseline


class V4GuardTests(unittest.TestCase):
    def test_sequence_windows_do_not_cross_scenarios(self):
        rows=[{'scenario_id':'a','ego_speed':'1','lead_speed':'1','distance':'1','relative_speed':'1','ttc':'1','ego_acceleration':'1','road_friction':'1','road_slope':'1','vehicle_mass':'1','reaction_time':'1','sensor_delay_ms':'1','brake_state':'0','risk_level':'Safe','stopping_distance':'1'} for _ in range(3)]
        rows += [{**rows[0],'scenario_id':'b'} for _ in range(3)]
        data=Sequences(rows,np.zeros(11),np.ones(11),['ego_speed','lead_speed','distance','relative_speed','ttc','ego_acceleration','road_friction','road_slope','vehicle_mass','reaction_time','sensor_delay_ms'],length=2)
        self.assertEqual(len(data),4)

    def test_delay_uses_only_previous_observation(self):
        x=torch.arange(2*4*11,dtype=torch.float32).reshape(2,4,11)
        out=perturbation('additional_delay',[str(i) for i in range(11)],np.ones(11))(x)
        self.assertTrue(torch.equal(out[:,1:,:],x[:,:-1,:]))

    def test_temporal_models_emit_multitask_outputs(self):
        x=torch.randn(2,10,11)
        for model in (TCNCausalBaseline(11),TransformerCausalBaseline(11)):
            logits,distance=model(x);self.assertEqual(tuple(logits.shape),(2,3));self.assertEqual(tuple(distance.shape),(2,))

    def test_causal_convolution_cannot_see_future(self):
        layer=CausalConv1d(1,1,3,dilation=2); layer.eval()
        x=torch.randn(1,1,10); changed=x.clone(); changed[:,:,7:]+=100.
        self.assertTrue(torch.allclose(layer(x)[:,:,:7],layer(changed)[:,:,:7]))

    def test_audit_preserves_scenario_split_and_train_scaler_rule(self):
        _,summary=audit('without_brake_state')
        self.assertEqual(summary['scenario_overlap'],0)
        self.assertEqual(summary['scaler_fit'],'train only')
        self.assertEqual(summary['forbidden_features_found'],[])


if __name__=='__main__':unittest.main()
