import React from 'react';
import WaveSurfer from 'wavesurfer.js';
import CursorPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.cursor';
import RegionsPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.regions';
import MinimapPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.minimap';
import TimelinePlugin from 'wavesurfer.js/dist/plugin/wavesurfer.timeline';
import './styles.css';
import Collapse from '@material-ui/core/Collapse';
import {
    PlayArrow,
    Pause,
    SyncDisabledOutlined,
    SyncOutlined,
} from '@material-ui/icons'

export default class ShowWave extends React.Component{
    constructor(props){
        super(props);
        this.state = {
            buttonIsPlay: false,
            buttonIsLoop: false,
            buttonIsShowRegions: true,
            buttonIsShowMiniMap: true,
            buttonIsShowTimeline: true,
            buttonIsShowCursor: true,
            checked_1: false,
            checked_2: false,
        }
    }
    initWavesurfer = () => {
        const options = {
            barWidth: 1,
            cursorWidth: 1,
            container: '#waveform',
            backend: 'MediaElement',
            height: 200,
            progressColor: '#005266',
            responsive: true,
            waveColor: '#fff',
            cursorColor: '#005266',
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
        // wavesurfer.on('ready', this.play)
    }
    onPlayEnd = () => {
        this.state.buttonIsLoop ? this.wavesurfer.play() : this.wavesurfer.play(this.region.start)
    }
    createRegions = () => {
        fetch("http://192.168.50.225:5000/getoutput/" + this.props.file)
            .then((response) => {
                let data_promise = Promise.resolve(response.json())
                data_promise.then((data) => {
                    console.log(data)
                    return data
                }).then((jsonData) => {
                    let len = jsonData.length
                    for (let i = 0; i < len; i++){
                        console.log(jsonData[i])
                        this.wavesurfer.addRegion(jsonData[i]) 
                    }
                    // let forEach = jsonData.forEach((start, end, type, drag, resize) => {
                    //     console.log(start, end, type, drag, resize)
                    //     this.wavesurfer.addRegion({
                    //         start: start,
                    //         end: end,
                    //         type: type,
                    //         drag: drag,
                    //         resize: resize,
                    //     })           
                    // })
                })
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
            color: "#005266",
            customShowTimeStyle: {
                'background-color': '#005266',
                color: '#fff',
                padding: '3px',
                'font-size': '15px'
            }
        })).initPlugin('cursor')
    }
    componentDidMount(){
        console.log(this.props.file)
        const url = "http://192.168.50.225:5000/wav/" + String(this.props.file)
        console.log(url)
        this.initWavesurfer()
        this.wavesurfer.load(url)
        
        var that = this;
		setTimeout(() => {
			that.show();
		}, that.props.wait);
    }
    show = () => {
        var that = this
		var delay = (s) => {
            return new Promise((resolve, reject) => {
                setTimeout(resolve,s); 
            });
        };
        delay().then(() => {
            that.setState({checked_1: true})
            return delay(800); 
        }).then(() => {
            that.setState({checked_2: true})
        });
    }
    togglePlay = () => {
        const bool = this.state.buttonIsPlay
        this.wavesurfer.playPause();
        this.setState({ buttonIsPlay: !bool })
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
                <Collapse in={this.state.checked_1} timeout={1300}> 
                    <div className="waveCursorContainer">
                        <div className="waveContainer">
                            <div id='waveform'></div>
                            <br/>
                            <div id='wave-minimap'></div>
                            <div id='wave-timeline'></div>
                        </div>
                    </div>
                </Collapse>
                <Collapse in={this.state.checked_2} timeout={2000}>
                    <div className="buttonContainer">
                        <button onClick={this.togglePlay}>
                            <div className="buttonIcon">
                                { this.state.buttonIsPlay ? <Pause/> : <PlayArrow/> }
                            </div>
                        </button>
                        <button onClick={this.toggleLoop}>
                            <div className="buttonIcon">
                                { this.state.buttonIsLoop ? <SyncOutlined/> : <SyncDisabledOutlined/> }
                            </div>
                        </button>
                        <button onClick={this.toggleClearRegions}>
                            { this.state.buttonIsShowRegions ? "disable regions" : "able regions" }
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
                    </div>
                </Collapse>
            </div>
        );
    }
}