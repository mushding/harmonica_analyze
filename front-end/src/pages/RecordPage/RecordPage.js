import React from 'react'
const recordedChunks = []
export default class RecordPage extends React.Component{
    constructor(props){
        super(props)
        this.state = {
            isStop: false,
        }
    }
    componentDidMount(){

    }
    toggleStart = () => {
        navigator.mediaDevices.getUserMedia({ audio: true, video: false })
            .then((stream) => {
                const options = {mimeType: 'audio/webm'};
                const mediaRecorder = new MediaRecorder(stream, options)
                console.log("record")
                mediaRecorder.ondataavailable = this.handleDataAvailable;
                
                // mediaRecorder.addEventListener("stop", () => {
                //     console.log("stop")
                // })
                mediaRecorder.start()
                setTimeout(() => {
                    mediaRecorder.stop()
                }, 3000)
            })
    }
    handleDataAvailable = (event) => {
        console.log("push")
        console.log(event.data)
        console.log(URL.createObjectURL(event.data))
        if (event.data.size > 0){
            recordedChunks.push(event.data);
        }
    }
    toggleStop = () => {
        console.log("press")
        this.setState({ isStop: true })
    }
    render(){
        return(
            <div>
                <button onClick={this.toggleStart}>START</button>
                <button onClick={this.toggleStop}>STOP</button>
            </div>
        );
    }
}