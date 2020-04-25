import React from 'react'
import Navbar from '../../components/Navbar/Navbar'
import NoteUploader from '../../components/NoteUploader/NoteUploader'

export default class NoteTestPage extends React.Component{
    componentDidMount(){
    }
    render(){
        return(
            <div>
                <Navbar/>
                <NoteUploader/>
            </div>
        )
    }
}