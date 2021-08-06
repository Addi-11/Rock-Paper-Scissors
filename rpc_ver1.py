{"cells":[{"cell_type":"code","execution_count":0,"outputs":[],"metadata":{"collapsed":false,"_kg_hide-input":false},"source":"\n\n\nimport os\nfor dirname, _, filenames in os.walk('/kaggle/input'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n        \n\nimport pandas as pd\nimport random\nimport numpy as np\nfrom kaggle_environments.envs.rps.utils import get_score\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom kaggle_environments import make, evaluate\nimport keras\nimport collections\nimport sys\nimport os\n\n# Util method for getting the state in the q table\ndef get_state(action, op_action):\n    return action * 3 + op_action\n# Current action\ncur_action = 0\n# Epsilon: Exploration rate\neps = 0.1\n# History\nhistory = []\n# Q_table (Policies) Shape: (9, 3)\npolicies = [[0] *3] * 9\n#Learning rate\nlr = 0.7\n# Discount rate for q_table\ndiscount_rate = 0.3\n# Epsilon decay rate\ndecay_rate = 0.9\n\ndef update_q_table(op_action):\n    global policies\n    global discount_rate\n    global lr\n    global history\n    reward = get_score(cur_action, op_action)\n    if len(history) > 1:\n        previous_state_id = get_state(history[len(history) - 2][0], history[len(history) - 2][1])\n        state_id = get_state(cur_action, op_action)\n        policies[previous_state_id][cur_action] = policies[previous_state_id][cur_action] * (1 - lr) \\\n        + lr * (reward + discount_rate * np.max(policies[state_id][:]))\n\ndef mdp(observation, configuration):\n    global cur_action\n    global history\n    global policies\n    if observation.step > 0:\n        history.append([cur_action, observation.lastOpponentAction])\n        update_q_table(observation.lastOpponentAction)\n    \n    explore_rate = np.random.random()\n    if explore_rate < eps:\n        cur_action = random.randint(0, 2)\n        explore_rate *= decay_rate\n    else:\n        if observation.step > 0:\n            state_id = get_state(cur_action, observation.lastOpponentAction)\n            cur_action = int(np.argmax(policies[state_id][:]))\n        else:\n            cur_action = random.randint(0, 2)\n    return cur_action        \n        \n        \nenv = make(\"rps\", configuration={\"episodeSteps\": 1000}, debug=\"True\")\nenv.reset()\nenv.run([\"mdp.py\", \"statistical\"])\nenv.render(mode=\"ipython\", width=400, height=400)\n"}],"metadata":{"language_info":{"name":"python","version":"3.6.6","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"},"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat":4,"nbformat_minor":4}