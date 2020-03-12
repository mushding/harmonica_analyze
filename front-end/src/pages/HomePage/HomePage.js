import React, { Component } from 'react';
import WaveSurfer from 'wavesurfer.js';

export default class HomePage extends Component{
    constructor(props){
        super(props);
    }
    componentDidMount(){
        const aud = document.querySelector('#song');
        this.waveSurfer = WaveSurfer.create({     
            barWidth: 1,
            cursorWidth: 1,
            container: '#waveform',
            backend: 'MediaElement',
            height: 200,
            progressColor: '#4a74a5',
            responsive: true,
            waveColor: '#ccc',
            cursorColor: '#4a74a5',
        });
        // this.waveSurfer.load("https://reelcrafter-east.s3.amazonaws.com/aux/test.m4a");
        // this.waveSurfer.load("http://192.168.50.225:5000/test/flat_normal_double.wav", [0.5, 0.6, 0.7, 0.1, 0.2, 0.2888])
        this.waveSurfer.load(require("./flat_normal_double.wav"))
    }
    playIt = () => {
        this.waveSurfer.playPause();
    }
    render(){
        return (
            <div>
                <button onClick={this.playIt}>
                    Play
                </button>
                <div id='waveform'>
                </div>
            </div>
        );
    }
}