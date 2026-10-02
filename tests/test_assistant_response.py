import unittest
from unittest.mock import patch
from src.assistant_response import generate_assistant_reply


class AssistantResponseTests(unittest.TestCase):
    def test_calculator_is_labelled_and_does_not_use_model(self):
        with patch('src.assistant_response.generate_chat_reply') as model:
            self.assertEqual(generate_assistant_reply(None,None,'What is 7+8?'),('15.',0,'local_calculator'))
            model.assert_not_called()

    def test_raw_model_option_is_respected(self):
        with patch('src.assistant_response.generate_chat_reply',return_value=('12.',80)) as model:
            self.assertEqual(generate_assistant_reply(None,None,'What is 7+8?',use_tools=False),('12.',80,'model'))
            model.assert_called_once()

    def test_nonmath_question_reaches_model_unchanged(self):
        with patch('src.assistant_response.generate_chat_reply',return_value=('A function is reusable code.',80)) as model:
            result=generate_assistant_reply(None,None,'Explain a Python function.')
            self.assertEqual(result[2],'model')
            self.assertEqual(model.call_args.args[2],'Explain a Python function.')


if __name__=='__main__':
    unittest.main()
