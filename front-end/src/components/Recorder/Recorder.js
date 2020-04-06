import React from "react";
import ShowWave from "../ShowWave/ShowWave";
import Collapse from "@material-ui/core/Collapse";
import "./styles.css";

// global varible area
let mediaRecorder = null;

export default class Recorder extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            isStop: false,
            checked: false,
            recordChecked: false,
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
                mediaRecorder.addEventListener('dataavailable', (e) => {
                    if (e.data.size > 0){
                        this.handleDataAvailable(e) 
                    }
                })
                mediaRecorder.start()
                this.setState({ recordChecked: true })
            })
    };
    handleDataAvailable = (event) => {
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
        this.handleDataAvailable()
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
                    </Collapse>
                    <Collapse in={this.state.recordChecked} timeout={2000}>
                        <div className="recordingTextContainer">
                            <h3>Recording ...</h3>
                        </div>
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
