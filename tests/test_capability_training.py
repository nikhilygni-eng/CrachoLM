import math
import unittest
import torch
from capability_curriculum import make_curriculum, normalize
from improve_capabilities import encode_example, collate, example_loss
from src.chat_response import chat_prompt
from src.tokenizer import CrachoTokenizer


class CapabilityTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.row={'prompt':'Hi','answer':'Hello!'}
        self.tok=CrachoTokenizer()
        self.tok.train_from_text(chat_prompt('Hi')+'Hello!')

    def test_prompt_and_padding_are_masked_answer_and_eos_are_not(self):
        x,y=encode_example(self.row,self.tok,128)
        start=len(self.tok.encode(chat_prompt('Hi')))-1
        expected=self.tok.encode('Hello!')+[self.tok.eos_id]
        self.assertEqual(y[:start].tolist(),[0]*start)
        self.assertEqual(y[start:].tolist(),expected)
        batch_x,batch_y=collate([(x,y),(x[:-2],y[:-2])])
        self.assertTrue(torch.all(batch_y[1,-2:]==0))

    def test_never_truncate_the_answer_or_end_marker(self):
        with self.assertRaises(ValueError):
            encode_example(self.row,self.tok,3)

    def test_answer_loss_ignores_prompt_and_padding(self):
        logits=torch.zeros(2,5,7)
        targets=torch.tensor([[0,0,2,3,0],[0,2,3,4,5]])
        self.assertAlmostEqual(example_loss(logits,targets).item(),math.log(7),places=6)
        logits[0,:2,0]=100
        logits[0,4,0]=100
        self.assertAlmostEqual(example_loss(logits,targets).item(),math.log(7),places=6)

    def test_task_groups_and_prompts_are_disjoint(self):
        rows=make_curriculum()
        for cat in ['grammar','arithmetic','reading','comparison']:
            groups={s:{r['group'] for r in rows if r['category']==cat and r['split']==s} for s in ['train','validation','test']}
            self.assertTrue(all(groups.values()))
            self.assertFalse(groups['train'] & groups['test'])
            self.assertFalse(groups['train'] & groups['validation'])
            self.assertFalse(groups['test'] & groups['validation'])
        prompts={s:{normalize(r['prompt']) for r in rows if r['split']==s} for s in ['train','validation','test']}
        self.assertFalse(prompts['train'] & prompts['test'])
        self.assertFalse(prompts['train'] & prompts['validation'])

    def test_arithmetic_labels_are_calculated_correctly(self):
        for row in make_curriculum():
            if row['category']!='arithmetic' or row['prompt'].startswith('Sam'):
                continue
            _,op,a,b=row['group'].split(':')
            a,b=int(a),int(b)
            expected=a+b if op=='+' else a-b if op=='-' else a*b
            self.assertEqual(row['answer'],str(expected)+'.')


if __name__=='__main__':
    unittest.main()
