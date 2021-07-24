/* 
*   Handy Cam
*   Copyright (c) 2021 Yusuf Olokoba.
*/

namespace NatSuite.Examples {

    using UnityEngine;
    using UnityEngine.UI;
    using NatSuite.Devices;
    using NatSuite.ML;
    using NatSuite.ML.Features;
    using NatSuite.ML.Vision;
    using NatSuite.ML.Visualizers;

    public class HandyCam : MonoBehaviour {
        
        [Header(@"NatML")]
        public string accessKey;

        [Header(@"Visualization")]
        public RawImage rawImage;
        public AspectRatioFitter aspectFitter;
        public HandPoseVisualizer visualizer;

        CameraDevice cameraDevice;
        Texture2D previewTexture;
        MLModelData modelData;
        MLModel model;
        HandPosePredictor predictor;

        async void Start () {
            // Request camera permissions
            if (!await MediaDeviceQuery.RequestPermissions<CameraDevice>()) {
                Debug.LogError(@"User did not grant camera permissions");
                return;
            }
            // Get the default camera device
            var query = new MediaDeviceQuery(MediaDeviceCriteria.CameraDevice);
            cameraDevice = query.current as CameraDevice;
            // Start the camera preview
            cameraDevice.previewResolution = (1280, 720);
            previewTexture = await cameraDevice.StartRunning();
            // Display the preview
            rawImage.texture = previewTexture;
            aspectFitter.aspectRatio = (float)previewTexture.width / previewTexture.height;
            // Fetch the model data from Hub
            Debug.Log("Fetching model data from Hub");
            modelData = await MLModelData.FromHub("@natsuite/hand-pose", accessKey);
            // Deserialize the model
            model = modelData.Deserialize();
            // Create the hand pose predictor
            predictor = new HandPosePredictor(model);
        }

        void Update () {
            // Check that predictor has been created
            if (predictor == null)
                return;
            // Predict the hand pose
            var hand = predictor.Predict(previewTexture);
            // Visualize
            visualizer.Render(hand);
        }

        void OnDisable () {
            // Dispose the model
            model?.Dispose();
            // Stop the camera preview
            if (cameraDevice?.running ?? false)
                cameraDevice.StopRunning();
        }
    }
}