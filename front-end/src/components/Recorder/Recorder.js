import React from "react";
import ShowWave from "../ShowWave/ShowWave";
import Collapse from "@material-ui/core/Collapse";
import "./styles.css";

// global varible area
let mediaRecorder = null;
let stream = null;
let recorder = null;
let recording = false;
let volume = null;
let audioInput = null;
let sampleRate = null;
let AudioContext = window.AudioContext || window.webkitAudioContext;
let context = null;
let analyser = null;
let tested = false;

export default class Recorder extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            isStop: false,
            checked: false,
            blobUrl: "",
            isMicAvailable: false,
        };
        this.micDeviceRef = React.createRef()
        this.canvasRef = React.createRef()
    }
    componentDidMount() {
        this.getMicDevice()
        var that = this;
        setTimeout(() => {
            that.show();
        }, that.props.wait);
    }
    getMicDevice = () => {
        navigator.mediaDevices.enumerateDevices()
            .then((deviceInfos) => {
                var mics = [];
                for (let i = 0; i !== deviceInfos.length; ++i) {
                    let deviceInfo = deviceInfos[i];
                    if (deviceInfo.kind === "audioinput") {
                        this.setState({ isMicAvailable: true })
                        mics.push(deviceInfo);
                        let label = deviceInfo.label || "Microphone " + mics.length;
                        console.log("Mic ", label + " " + deviceInfo.deviceId);
                        const option = document.createElement("option");
                        option.value = deviceInfo.deviceId;
                        option.text = label;
                        this.micDeviceRef.current.appendChild(option);
                    }
                }
            })
    }
    drawSineWave = () => {
        let canvasCtx = this.canvasRef.current.getContext("2d");

        let WIDTH = this.canvasRef.current.width;
        let HEIGHT = this.canvasRef.current.height;
        let CENTERX = this.canvasRef.current.width / 2;
        let CENTERY = this.canvasRef.current.height / 2;

        analyser.fftSize = 2048;
        var bufferLength = analyser.fftSize;
        console.log(bufferLength);
        var dataArray = new Uint8Array(bufferLength);
        console.log(dataArray)

        canvasCtx.clearRect(0, 0, WIDTH, HEIGHT);

        var draw = () => {

            // drawVisual = requestAnimationFrame(draw);

            analyser.getByteTimeDomainData(dataArray);

            canvasCtx.fillStyle = 'rgb(200, 200, 200)';
            canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);

            canvasCtx.lineWidth = 2;
            canvasCtx.strokeStyle = 'rgb(0, 0, 0)';

            canvasCtx.beginPath();

            var sliceWidth = WIDTH * 1.0 / bufferLength;
            var x = 0;

            for (var i = 0; i < bufferLength; i++) {
                console.log(dataArray)
                var v = dataArray[i] / 128.0;
                var y = v * HEIGHT/2;

                if(i === 0) {
                    canvasCtx.moveTo(x, y);
                } else {
                    canvasCtx.lineTo(x, y);
                }

                x += sliceWidth;
            }

            canvasCtx.lineTo(WIDTH, CENTERY);
            canvasCtx.stroke();
        };

        draw();
    }
    show = () => {
        var that = this;
        var delay = s => {
            return new Promise((resolve, reject) => {
                setTimeout(resolve, s);
            });
        };
        delay().then(() => {
            delay(500);
            that.setState({ checked: true });
        });
    };
    toggleStart = () => {
        if (!this.state.isMicAvailable){
            alert("There are no mic device.")
            return
        }
        const contraints = {
            audio: {
                deviceId: {
                    exact: this.micDeviceRef.current.value,
                },
            },
            video: false,
        }
        navigator.mediaDevices.getUserMedia(contraints)
            .then(stream => {
                const options = {
                    audioBitsPerSecond: 44100 * 16,
                    mimeType: "audio/webm"
                };
                mediaRecorder = new MediaRecorder(stream, options);
                mediaRecorder.ondataavailable = this.handleDataAvailable;
                mediaRecorder.start();
                this.setUpRecording(stream)
            })
    };
    setUpRecording = (stream) => {
        context = new AudioContext();
        sampleRate = context.sampleRate;
        
        // creates a gain node
        volume = context.createGain();
        
        // creates an audio node from teh microphone incoming stream
        audioInput = context.createMediaStreamSource(stream);
        
        // Create analyser
        analyser = context.createAnalyser();
        
        // connect audio input to the analyser
        audioInput.connect(analyser);
        
        // connect analyser to the volume control
        // analyser.connect(volume);
        
        let bufferSize = 2048;
        let recorder = context.createScriptProcessor(bufferSize, 2, 2);
        
        // we connect the volume control to the processor
        // volume.connect(recorder);
        
        analyser.connect(recorder);
        
        // finally connect the processor to the output
        recorder.connect(context.destination); 
        this.drawSineWave();
    };
    handleDataAvailable = event => {
        const blobDataInWebaFormat = event.data;
        const blobDataInWavFormat: Blob = new Blob([blobDataInWebaFormat], { type: "audio/wav" });
        const dataUrl = URL.createObjectURL(blobDataInWavFormat);
        this.setState({ blobUrl: dataUrl });
        console.log(blobDataInWavFormat);
        console.log(dataUrl);

        // formdata
        let wavdata = new FormData();
        wavdata.append("data", blobDataInWavFormat);

        fetch("https://www.haranalyzer.site/recordUpload", {
            method: "POST",
            body: wavdata
        }).then(response => {
            this.setState({ isStop: true });
        });
    };
    toggleStop = () => {
        mediaRecorder.stop();
    };
    render() {
        if (!this.state.isStop) {
            return (
                <div>
                    <Collapse in={this.state.checked} timeout={2000}>
                        <div className="recordContainer">
                            <select className="selectContainer" ref={this.micDeviceRef} name="" id="micSelect"></select>
                            <button
                                className="recordButtonContainer"
                                onClick={this.toggleStart}
                            >
                                start record
                            </button>
                            <button
                                className="recordButtonContainer"
                                onClick={this.toggleStop}
                            >
                                stop record
                            </button>
                        </div>
                        <canvas width="500" height="300" ref={this.canvasRef}></canvas>
                    </Collapse>
                </div>
            );
        } else {
            return (
                <ShowWave
                    isRecord={true}
                    blobUrl={this.state.blobUrl}
                    file="record.wav"
                />
            );
        }
    }
}
