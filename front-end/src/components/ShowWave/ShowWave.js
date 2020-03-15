import React from 'react';
import WaveSurfer from 'wavesurfer.js';
import CursorPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.cursor';
import RegionsPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.regions';
import MinimapPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.minimap';
import TimelinePlugin from 'wavesurfer.js/dist/plugin/wavesurfer.timeline';
import './styles.css';

export default class ShowWave extends React.Component{
    constructor(props){
        super(props);
        this.state = {
            buttonIsShowRegions: true,
            buttonIsLoop: false,
            buttonIsShowMiniMap: true,
            buttonIsShowTimeline: true,
            buttonIsShowCursor: true,
        }
    }
    initWavesurfer = () => {
        const options = {
            barWidth: 1,
            cursorWidth: 1,
            container: '#waveform',
            backend: 'MediaElement',
            height: 200,
            progressColor: '#4a74a5',
            responsive: true,
            waveColor: '#ccc',
            cursorColor: '#4a74a5',
            plugins: [
                RegionsPlugin.create(),
            ]
        };
        const wavesurfer = WaveSurfer.create(options);
        this.wavesurfer = wavesurfer
        this.createRegions()
        this.createMinimap()
        this.createTimeline()
        this.createCursor()

        // event listener
        wavesurfer.on('region-in', (region) => {
            this.region = region
        })
        wavesurfer.on('region-out', this.onPlayEnd)
        wavesurfer.on('ready', this.play)
    }
    onPlayEnd = () => {
        this.state.buttonIsLoop ? this.wavesurfer.play() : this.wavesurfer.play(this.region.start)
    }
    createRegions = () => {
        fetch("http://192.168.50.225:5000/sendsec")
            .then((response) => {
                return response.json()
            })
            .then((jsonData) => {
                this.wavesurfer.addRegion(jsonData)           
            })
            .catch((error) => {
                console.log(error)
                return false
            })
        return true
    }
    createMinimap = () => {
        this.wavesurfer.addPlugin(MinimapPlugin.create({
            container: '#wave-minimap',
            waveColor: '#777',
            progressColor: '#222',
            height: 50,
        })).initPlugin('minimap')
    }
    createTimeline = () => {
        this.wavesurfer.addPlugin(TimelinePlugin.create({
            container: "#wave-timeline",
        })).initPlugin('timeline')
    }
    createCursor = () => {
        this.wavesurfer.addPlugin(CursorPlugin.create({
            showTime: true,
            opacity: 1,
            hideOnBlur: true,
            customShowTimeStyle: {
                'background-color': '#000',
                color: '#fff',
                padding: '2px',
                'font-size': '10px'
            }
        })).initPlugin('cursor')
    }
    componentDidMount(){
        this.initWavesurfer()
        this.wavesurfer.load(require("./double.wav"))
    }
    play = () => {
        this.wavesurfer.playPause();
    }
    toggleClearRegions = () => {
        const bool = this.state.buttonIsShowRegions
        bool ? this.wavesurfer.clearRegions() : this.createRegions()
        this.setState({ buttonIsShowRegions: !bool })
    }
    toggleLoop = () => {
        const bool = this.state.buttonIsLoop
        this.setState({ buttonIsLoop: !bool })
    }
    toggleShowMiniMap = () => {
        const bool = this.state.buttonIsShowMiniMap
        bool ? this.wavesurfer.destroyPlugin('minimap') : this.createMinimap()
        this.setState({ buttonIsShowMiniMap: !bool})
    }
    toggleShowTimeline = () => {
        const bool = this.state.buttonIsShowTimeline
        bool ? this.wavesurfer.destroyPlugin('timeline') : this.createTimeline()
        this.setState({ buttonIsShowTimeline: !bool})
    }
    toggleShowCursor = () => {
        const bool = this.state.buttonIsShowCursor
        bool ? this.wavesurfer.destroyPlugin('cursor') : this.createCursor()
        this.setState({ buttonIsShowCursor: !bool})
    }
    render(){
        return (
            <div>
                <button onClick={this.play}>
                    Play
                </button>
                <button onClick={this.toggleClearRegions}>
                    { this.state.buttonIsShowRegions ? "disable regions" : "able regions" }
                </button>
                <button onClick={this.toggleLoop}>
                    { this.state.buttonIsLoop ? "able loop" : "disable loop" }
                </button>
                <button onClick={this.toggleShowMiniMap}>
                    { this.state.buttonIsShowMiniMap ? "close minimap" : "open minimap" }
                </button>
                <button onClick={this.toggleShowTimeline}>
                    { this.state.buttonIsShowTimeline ? "close timeline" : "open timeline" }
                </button>
                <button onClick={this.toggleShowCursor}>
                    { this.state.buttonIsShowCursor ? "close cursor" : "open sursor"}
                </button>
                <div class="waveContainer">
                    <div id='waveform'></div>
                    <br/>
                    <div id='wave-minimap'></div>
                    <div id='wave-timeline'></div>
                </div>
            </div>
        );
    }
}