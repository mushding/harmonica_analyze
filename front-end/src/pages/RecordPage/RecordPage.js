import React from 'react'
import Navbar from '../../components/Navbar/Navbar'
import Recorder from '../../components/Recorder/Recorder'
export default class RecordPage extends React.Component{
    componentDidMount(){
    }
    render(){
        return(
            <div>
                <Navbar/>
                <Recorder/>
            </div>
        );
    }
}