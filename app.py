from flask import Flask, request, jsonify, Response, render_template
from flask_restful import Resource, Api
import pymongo

import json

import tensorflow as tf
import os

import tf_agents

import numpy as np
import ssl

try:
    mongo = pymongo.MongoClient("mongodb+srv://user:qB0xFmph33okh2D7@Curate.9jbnx.mongodb.net/Curate?retryWrites=true&w=majority",
        serverSelectionTimeoutMS = 1000)
    db = mongo.Curate
    db = db.curate
    monngo.server_info()
except:
    print("Cannot connect to db")


app = Flask(__name__)
api = Api(app)

container = []

class Predict(Resource):
    def get(self):

        return render_template('index.html')

    def post(self):

        mongo = pymongo.MongoClient("mongodb+srv://user:qB0xFmph33okh2D7@Curate.9jbnx.mongodb.net/Curate?retryWrites=true&w=majority",
            serverSelectionTimeoutMS = 1000)
        db = mongo.Curate
        db = db.curate

        policy_dir = "Perceptual Learning Model"
        model = tf.compat.v2.saved_model.load(policy_dir)



        data = request.get_json()
        id = data["participantId"]


        participant_folder = db

        """for logging of data"""

        if len(list(participant_folder.find({"participantId": id}))) == 0:

            participant_folder.find_one_and_update(
            {"participantId": id},
            {"$set":
            {"participantId": id,
            "data": [],
            }}, upsert = True
            )

        x = participant_folder.find({"participantId": id})
        x = list(x)

        new_data = x[0]["data"]
        # acc_container = x[0]["accuracy"]
        # reward_container = x[0]["reward"]
        # action_container = x[0]["action"]

        del data["participantId"]


        new_data.append(data)

        response_container = []

        for i in new_data:
            response_container.append(i["correct"])

        acc = sum(response_container)/len(response_container)
        acc2 = round(acc,3)

        # acc_container.append(acc)

        if acc2 == 0.8:
            reward = 100
        else:
            reward = round(-abs((0.8 - acc)*100),3)

        # reward_container.append(reward)


        """feeding data for RL_Agent and making prediction"""

    #observation = rotation, noise, accuracy, reward = current reward

        rotation= data["rotation"]
        noise = data["noise"]


        observation = tf.convert_to_tensor(np.array([rotation,noise,acc2]),np.float32)
        observation = tf.reshape(observation, [1,3])

        reward2 = tf.convert_to_tensor(np.array([reward]),np.float32)

        timestep = tf_agents.trajectories.time_step.transition(
            reward= reward2,
            observation=observation
            )

        act= model.action(timestep).action

        action = act.numpy()

        action = action.tolist()

        action = action[0]

        # action_container.append(action)

        """ Adding data to mongoDB"""

        new_data.pop(-1)

        data["accuracy"] = acc
        data["reward"] = reward
        data["action"] = action

        new_data.append(data)

        participant_folder.find_one_and_update(
        {"participantId": id},
            {"$set":
            {"data" : new_data,
            }}, upsert = True)


        return action

        # Response(response = json.dumps({"data" : f"{action_container}"}))




api.add_resource(Predict, "/")


if __name__ == "__main__":
    app.run(debug = False)
